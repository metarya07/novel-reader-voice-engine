#!/usr/bin/env python3
"""
Phase 1: Subaru Natsuki Voice Emotion Experiment (Pristine Fidelity)
===================================================================
Generates 10 WAV files: 5 emotions x (FLAT version + EMOTION version).
Voice Base: en-US-AndrewMultilingualNeural (Locked for 100% Sean Chiplock vocal purity)
Pipeline: Edge TTS -> RVC v2 (rmvpe + FAISS) -> Sigmoid Silence Gate -> Analog Warmth Saturation
NO vocoder phase warping, NO time-stretching, NO voice swapping.
"""

import os
import sys
import json
import io
import asyncio
import tempfile
import subprocess
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
import torch
import edge_tts

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

PYTHON_VENV      = r"a:\Projects\novel reader\voice-server\.venv\Scripts\python.exe"
RVC_ENGINE_DIR   = r"a:\Projects\novel reader\rvc-engine"
SUBARU_MODEL     = r"a:\Projects\novel reader\voice-server\models\subaru\subaru_e50_s22500.pth"
SUBARU_INDEX     = r"a:\Projects\novel reader\voice-server\models\subaru\subaru_added_IVF256_Flat_nprobe_1_subaru_v2.index"
PARAMS_FILE      = r"a:\Projects\novel-reader-voice-engine\params\subaru_params.json"
OUTPUT_DIR       = r"a:\Projects\novel-reader-voice-engine\experiments\subaru\output"
ARTIFACT_DIR     = r"C:\Users\metar\.gemini\antigravity\brain\d4a04e3c-caff-41c4-ab27-b0cb43df6115"

TEST_LINES = [
    (1, "rage",        "I'll save you, I swear it! No matter how many times it takes!"),
    (2, "grief",       "I'm sorry. I'm so sorry, Rem. I'm nothing. I'm worthless."),
    (3, "comedic",     "Wait, seriously?! That actually worked?! Ha! I'm a genius!"),
    (4, "cold_threat", "Don't come near her. If you take one more step, I will end you."),
    (5, "gratitude",   "Emilia... thank you. Just, thank you for believing in me."),
]

BASE_VOICE = "en-US-AndrewMultilingualNeural"

FLAT_PARAMS = {
    "voice":                BASE_VOICE,
    "edge_pitch_hz":        0,
    "edge_rate_pct":        0,
    "rvc_pitch_semitones":  2,
    "index_rate":           0.88,
    "rms_mix_rate":         0.20,
    "protect":              0.50,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. EDGE TTS GENERATION
# ─────────────────────────────────────────────────────────────

async def _tts_attempt(text: str, voice: str, pitch_str: str, rate_str: str) -> bytes:
    comm = edge_tts.Communicate(text=text, voice=voice, pitch=pitch_str, rate=rate_str)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    data = buf.getvalue()
    if len(data) < 200:
        raise RuntimeError("Edge TTS returned empty audio payload")
    return data


async def generate_tts(text: str, pitch_hz: int, rate_pct: int, voice: str = BASE_VOICE) -> bytes:
    pitch_str = f"{pitch_hz:+d}Hz"
    rate_str = f"{rate_pct:+d}%"
    print(f"     [TTS] voice={voice} pitch={pitch_str} rate={rate_str}")
    for attempt in range(1, 6):
        try:
            return await _tts_attempt(text, voice, pitch_str, rate_str)
        except Exception as exc:
            wait = 2 ** attempt
            if attempt < 5:
                print(f"     [TTS] Attempt {attempt} failed ({type(exc).__name__}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Edge TTS failed after 5 attempts: {exc}") from exc


# ─────────────────────────────────────────────────────────────
# 2. RVC v2 INFERENCE (CUDA, RMVPE, FAISS IVF256)
# ─────────────────────────────────────────────────────────────

def run_rvc_inference(raw_audio_bytes: bytes, pitch_semitones: int,
                      index_rate: float, rms_mix_rate: float = 0.20,
                      protect: float = 0.50, index_path: str = None) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
        in_path = fin.name
        out_path = fout.name
        fin.write(raw_audio_bytes)

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        py_exec = PYTHON_VENV if os.path.exists(PYTHON_VENV) else sys.executable
        cmd = [
            py_exec, "-m", "infer.cli",
            "--model",        SUBARU_MODEL,
            "--input",        in_path,
            "--output",       out_path,
            "--f0-method",    "rmvpe",
            "--pitch",        str(pitch_semitones),
            "--index-rate",   str(index_rate),
            "--rms-mix-rate", str(rms_mix_rate),
            "--protect",      str(protect),
            "--overwrite"
        ]
        target_index = index_path if (index_path and os.path.isfile(index_path)) else SUBARU_INDEX
        if os.path.isfile(target_index):
            cmd.extend(["--index", target_index])
            print(f"     [RVC] using index: {os.path.basename(target_index)}")

        print(f"     [RVC] pitch={pitch_semitones:+d}st index_rate={index_rate} rms_mix={rms_mix_rate} protect={protect}")
        res = subprocess.run(
            cmd, cwd=RVC_ENGINE_DIR,
            capture_output=True, text=True,
            encoding="utf-8", errors="ignore", env=env
        )
        if res.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 500:
            err = (res.stderr or res.stdout or "").strip()[:200]
            raise RuntimeError(f"RVC inference failed (code {res.returncode}): {err}")

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass


# ─────────────────────────────────────────────────────────────
# 3. SIGMOID SILENCE GATE
# ─────────────────────────────────────────────────────────────

def apply_silence_gate(source_bytes: bytes, converted_bytes: bytes) -> bytes:
    """
    Input-guided sigmoid gate.
    Mutes synthetic 226Hz vocoder hum during dialogue pauses based on source TTS energy.
    """
    y_src, sr_src = sf.read(io.BytesIO(source_bytes), dtype="float32")
    y_conv, sr_conv = sf.read(io.BytesIO(converted_bytes), dtype="float32")

    if y_src.ndim > 1: y_src = np.mean(y_src, axis=1)
    if y_conv.ndim > 1: y_conv = np.mean(y_conv, axis=1)

    if sr_src != sr_conv:
        y_src = librosa.resample(y_src, orig_sr=sr_src, target_sr=sr_conv)

    min_len = min(len(y_src), len(y_conv))
    y_src_sub = y_src[:min_len]
    y_conv_sub = y_conv[:min_len]

    win = int(0.025 * sr_conv)
    hop = int(0.010 * sr_conv)
    rms_in = np.zeros(min_len, dtype=np.float32)

    for i in range(0, min_len - win, hop):
        rms_in[i:i + hop] = np.sqrt(np.mean(y_src_sub[i:i + win] ** 2))

    sigma = max(1, int(0.025 * sr_conv / hop))
    rms_smooth = gaussian_filter1d(rms_in, sigma=sigma)
    gate = 1.0 / (1.0 + np.exp(-(rms_smooth - 0.0015) / 0.0003))

    gated = y_conv_sub * gate
    out_buf = io.BytesIO()
    sf.write(out_buf, gated, sr_conv, format="WAV")
    return out_buf.getvalue()


# ─────────────────────────────────────────────────────────────
# 4. ANALOG WARMTH SATURATION (Subtle chest resonance)
# ─────────────────────────────────────────────────────────────

def apply_warmth_saturation(audio_bytes: bytes) -> bytes:
    audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    tensor = torch.from_numpy(audio_data)
    max_val = torch.max(torch.abs(tensor))
    if max_val > 0.001:
        tensor = tensor / max_val * 0.95
    tensor = torch.tanh(tensor * 1.08) / 1.03

    out_buf = io.BytesIO()
    sf.write(out_buf, tensor.numpy(), sr, format="WAV")
    return out_buf.getvalue()


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────

def process_and_save(text: str, cfg: dict, label: str, out_path: str):
    print(f"  -> {label:10s} generating...")
    # 1. Edge TTS
    raw_tts = asyncio.run(generate_tts(
        text=text,
        pitch_hz=cfg["edge_pitch_hz"],
        rate_pct=cfg["edge_rate_pct"],
        voice=cfg.get("voice", BASE_VOICE)
    ))

    # 2. RVC with Emotion-Dedicated Index
    idx_file = cfg.get("index_file", "")
    idx_path = None
    if idx_file:
        candidate = os.path.join(r"a:\Projects\novel-reader-voice-engine\models\subaru\indices", idx_file)
        if os.path.isfile(candidate):
            idx_path = candidate

    rvc_out = run_rvc_inference(
        raw_audio_bytes=raw_tts,
        pitch_semitones=cfg["rvc_pitch_semitones"],
        index_rate=cfg["index_rate"],
        rms_mix_rate=cfg.get("rms_mix_rate", 0.20),
        protect=cfg.get("protect", 0.50),
        index_path=idx_path
    )

    # 3. Silence Gate
    gated = apply_silence_gate(raw_tts, rvc_out)

    # 4. Analog Warmth
    final_wav = apply_warmth_saturation(gated)

    with open(out_path, "wb") as f:
        f.write(final_wav)

    size_kb = len(final_wav) // 1024
    print(f"  [OK] {label:10s} -> {os.path.basename(out_path)} ({size_kb} KB)")


def main():
    print("=" * 70)
    print("  SUBARU VOICE EMOTION EXPERIMENT — PRISTINE VOCAL FIDELITY")
    print(f"  Base Voice: {BASE_VOICE} (Locked)")
    print("  Pipeline: Edge TTS -> RVC v2 -> Silence Gate -> Analog Saturation")
    print("=" * 70)

    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        all_params = json.load(f)

    generated = []

    for idx, emotion, text in TEST_LINES:
        idx_str = f"{idx:02d}"
        flat_path = os.path.join(OUTPUT_DIR, f"subaru_exp_{idx_str}_{emotion}_FLAT.wav")
        emo_path = os.path.join(OUTPUT_DIR, f"subaru_exp_{idx_str}_{emotion}_EMOTION.wav")
        emo_cfg = all_params.get(emotion, FLAT_PARAMS)

        print(f"\n[{idx_str}/05] {emotion.upper()}")
        print(f'      "{text}"')

        # Generate FLAT
        process_and_save(text, FLAT_PARAMS, "FLAT", flat_path)

        # Generate EMOTION
        process_and_save(text, emo_cfg, "EMOTION", emo_path)

        generated.extend([flat_path, emo_path])

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT COMPLETE: {len(generated)}/10 WAVs generated cleanly")
    print("=" * 70)
    for p in generated:
        print(f"  {os.path.basename(p):55s} {os.path.getsize(p)//1024:>5} KB")

    print("\n[Copying to artifact directory...]")
    copied = 0
    for p in generated:
        dest = os.path.join(ARTIFACT_DIR, os.path.basename(p))
        try:
            shutil.copy2(p, dest)
            copied += 1
        except Exception as e:
            print(f"  [WARN] {os.path.basename(p)}: {e}")
    print(f"[OK] {copied}/{len(generated)} WAVs copied to artifacts")


if __name__ == "__main__":
    main()
