#!/usr/bin/env python3
"""
Phase 1: Subaru Natsuki Voice Experiment Pipeline
Generates 10 WAV files: 5 emotions x (FLAT version + EMOTION version).
Voice Base: en-US-AndrewMultilingualNeural, rms_mix_rate=0.20, Analog Warmth Saturation.
"""

import os
import sys
import json
import io
import asyncio
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
import torch
import edge_tts

# Absolute Paths
PYTHON_VENV = r"a:\Projects\novel reader\voice-server\.venv\Scripts\python.exe"
RVC_ENGINE_DIR = r"a:\Projects\novel reader\rvc-engine"
SUBARU_MODEL_PATH = r"a:\Projects\novel reader\voice-server\models\subaru\subaru_e50_s22500.pth"
SUBARU_INDEX_PATH = r"a:\Projects\novel reader\voice-server\models\subaru\subaru_added_IVF256_Flat_nprobe_1_subaru_v2.index"
OUTPUT_DIR = r"a:\Projects\novel-reader-voice-engine\experiments\subaru\output"
PARAMS_FILE = r"a:\Projects\novel-reader-voice-engine\params\subaru_params.json"

TEST_LINES = [
    ("rage", "I'll save you, I swear it! No matter how many times it takes!"),
    ("grief", "I'm sorry. I'm so sorry, Rem. I'm nothing. I'm worthless."),
    ("comedic", "Wait — seriously?! That actually worked?! Ha! I'm a genius!"),
    ("cold_threat", "Don't come near her. If you take one more step, I will end you."),
    ("gratitude", "Emilia… thank you. Just — thank you for believing in me."),
]

BASE_VOICE = "en-US-AndrewMultilingualNeural"


def apply_silence_gate(source_bytes, converted_bytes):
    from scipy.ndimage import gaussian_filter1d
    import soundfile as sf, numpy as np, io
    y_src, sr_src = sf.read(io.BytesIO(source_bytes), dtype="float32")
    y_conv, sr_conv = sf.read(io.BytesIO(converted_bytes), dtype="float32")
    if y_src.ndim > 1: y_src = np.mean(y_src, axis=1)
    if y_conv.ndim > 1: y_conv = np.mean(y_conv, axis=1)
    min_len = min(len(y_src), len(y_conv))
    y_src_s = y_src[:min_len]; y_conv_s = y_conv[:min_len]
    win = int(0.025 * sr_conv); hop = int(0.010 * sr_conv)
    rms = np.zeros(min_len, dtype=np.float32)
    for i in range(0, min_len - win, hop):
        rms[i:i+hop] = np.sqrt(np.mean(y_src_s[i:i+win]**2))
    rms_s = gaussian_filter1d(rms, sigma=int(0.025 * sr_conv / hop))
    gate = 1.0 / (1.0 + np.exp(-(rms_s - 0.0015) / 0.0003))
    out = io.BytesIO()
    sf.write(out, y_conv_s * gate, sr_conv, format="WAV")
    return out.getvalue()


async def generate_tts(text: str, pitch_str: str, rate_str: str, voice: str = BASE_VOICE) -> bytes:
    """Generate raw audio stream via edge-tts with retry on transient network errors."""
    import time
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            comm = edge_tts.Communicate(text=text, voice=voice, pitch=pitch_str, rate=rate_str)
            raw_stream = io.BytesIO()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    raw_stream.write(chunk["data"])
            data = raw_stream.getvalue()
            if len(data) < 100:
                raise RuntimeError("Edge TTS returned empty/tiny audio payload")
            return data
        except Exception as exc:
            wait = 2 ** attempt  # 2, 4, 8, 16, 32 seconds
            if attempt < max_attempts:
                print(f"     [TTS] Attempt {attempt} failed ({type(exc).__name__}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Edge TTS failed after {max_attempts} attempts: {exc}") from exc


def run_rvc_inference(
    raw_audio_bytes: bytes,
    pitch_semitones: int,
    index_rate: float,
    rms_mix_rate: float = 0.20,
    protect: float = 0.50
) -> bytes:
    """Run RVC v2 inference CLI via subprocess."""
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
            "--model", SUBARU_MODEL_PATH,
            "--input", in_path,
            "--output", out_path,
            "--f0-method", "rmvpe",
            "--pitch", str(pitch_semitones),
            "--index-rate", str(index_rate),
            "--rms-mix-rate", str(rms_mix_rate),
            "--protect", str(protect),
            "--overwrite"
        ]
        if os.path.exists(SUBARU_INDEX_PATH):
            cmd.extend(["--index", SUBARU_INDEX_PATH])

        res = subprocess.run(
            cmd,
            cwd=RVC_ENGINE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env
        )
        if res.returncode != 0:
            err = res.stderr.strip() or res.stdout.strip()
            raise RuntimeError(f"RVC inference failed (code {res.returncode}): {err}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            raise RuntimeError(f"RVC output file empty or missing: {out_path}")

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def prepare_source_for_gate(source_bytes: bytes, target_sr: int) -> bytes:
    """Resample TTS source bytes to match converted sample rate so silence gate operates at 1:1 timeline."""
    y, sr = sf.read(io.BytesIO(source_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    out = io.BytesIO()
    sf.write(out, y, target_sr, format="WAV")
    return out.getvalue()


def apply_warmth_saturation(audio_bytes: bytes) -> bytes:
    """Apply gentle analog warmth saturation on audio to enhance low-mid chest resonance and body."""
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


def process_sentence(
    text: str,
    pitch_str: str,
    rate_str: str,
    rvc_pitch: int,
    index_rate: float,
    rms_mix_rate: float = 0.20,
    protect: float = 0.50,
    voice: str = BASE_VOICE
) -> bytes:
    """Full pipeline: edge_tts -> RVC inference -> silence gate -> warmth saturation."""
    # 1. Edge TTS
    raw_tts_bytes = asyncio.run(generate_tts(text, pitch_str, rate_str, voice=voice))

    # 2. RVC Inference
    rvc_output_bytes = run_rvc_inference(
        raw_audio_bytes=raw_tts_bytes,
        pitch_semitones=rvc_pitch,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect=protect
    )

    # 3. Silence Gate
    _, conv_sr = sf.read(io.BytesIO(rvc_output_bytes))
    source_aligned = prepare_source_for_gate(raw_tts_bytes, conv_sr)
    gated_wav_bytes = apply_silence_gate(source_aligned, rvc_output_bytes)

    # 4. Analog Warmth Saturation
    final_wav_bytes = apply_warmth_saturation(gated_wav_bytes)
    return final_wav_bytes


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        params = json.load(f)

    generated_files = []

    print("================================================================================")
    print("Subaru Voice Experiment — Character-Depth Tuning (Sample 5 Profile)")
    print(f"Base Voice: {BASE_VOICE} | RMS Mix: 0.20 | Analog Saturation: Active")
    print("================================================================================")

    for i, (emotion, text) in enumerate(TEST_LINES, start=1):
        idx_str = f"{i:02d}"
        flat_filename = f"subaru_exp_{idx_str}_{emotion}_FLAT.wav"
        emotion_filename = f"subaru_exp_{idx_str}_{emotion}_EMOTION.wav"

        flat_path = os.path.join(OUTPUT_DIR, flat_filename)
        emotion_path = os.path.join(OUTPUT_DIR, emotion_filename)

        print(f"\n[{idx_str}/05] Emotion: {emotion.upper()}")
        print(f"      Text: \"{text}\"")

        # 1. Generate FLAT version
        if os.path.exists(flat_path) and os.path.getsize(flat_path) > 1000:
            print(f"  -> FLAT already exists, skipping: {flat_path}")
            generated_files.append((emotion, "FLAT", flat_path))
        else:
            # FLAT uses pitch="+0Hz", rate="+0%", rvc_pitch=2, index_rate=0.88, rms_mix=0.20
            print(f"  -> Generating FLAT version: pitch=+0Hz, rate=+0%, rvc_pitch=2, index_rate=0.88, rms_mix=0.20...")
            flat_wav = process_sentence(
                text=text,
                pitch_str="+0Hz",
                rate_str="+0%",
                rvc_pitch=2,
                index_rate=0.88,
                rms_mix_rate=0.20,
                protect=0.50,
                voice=BASE_VOICE
            )
            with open(flat_path, "wb") as f:
                f.write(flat_wav)
            print(f"     Saved: {flat_path} ({len(flat_wav):,} bytes)")
            generated_files.append((emotion, "FLAT", flat_path))

        # 2. Generate EMOTION version
        emo_cfg = params.get(emotion, params.get("neutral", {}))
        edge_pitch = emo_cfg.get("edge_pitch_hz", 0)
        edge_rate = emo_cfg.get("edge_rate_pct", 0)
        rvc_pitch = emo_cfg.get("rvc_pitch_semitones", 2)
        idx_rate = emo_cfg.get("index_rate", 0.85)
        rms_mix = emo_cfg.get("rms_mix_rate", 0.20)
        protect = emo_cfg.get("protect", 0.50)
        voice = emo_cfg.get("voice", BASE_VOICE)

        pitch_str = f"{edge_pitch:+d}Hz"
        rate_str = f"{edge_rate:+d}%"

        if os.path.exists(emotion_path) and os.path.getsize(emotion_path) > 1000:
            print(f"  -> EMOTION already exists, skipping: {emotion_path}")
            generated_files.append((emotion, "EMOTION", emotion_path))
        else:
            print(f"  -> Generating EMOTION version: pitch={pitch_str}, rate={rate_str}, rvc_pitch={rvc_pitch}, index_rate={idx_rate}, rms_mix={rms_mix}...")
            emotion_wav = process_sentence(
                text=text,
                pitch_str=pitch_str,
                rate_str=rate_str,
                rvc_pitch=rvc_pitch,
                index_rate=idx_rate,
                rms_mix_rate=rms_mix,
                protect=protect,
                voice=voice
            )
            with open(emotion_path, "wb") as f:
                f.write(emotion_wav)
            print(f"     Saved: {emotion_path} ({len(emotion_wav):,} bytes)")
            generated_files.append((emotion, "EMOTION", emotion_path))

    print("\n================================================================================")
    print("Experiment Complete! Generated 10 Audio Files with Sample 5 Tone & Depth:")
    print("================================================================================")
    for emotion, variant, path in generated_files:
        norm_path = Path(path).as_posix()
        uri = f"file:///{norm_path}"
        print(f" - [{variant:7s}] {emotion:12s}: {uri}")


if __name__ == "__main__":
    main()
