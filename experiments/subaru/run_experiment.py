#!/usr/bin/env python3
"""
Phase 1 -- Subaru Natsuki Emotion Experiment (v2 -- Best Accuracy)
=================================================================
Three-tier expressiveness strategy:

  Tier 1: Per-emotion Edge TTS voice selection
          (DavisNeural=rage, TonyNeural=grief, ChristopherNeural=comedic,
           GuyNeural=cold_threat, AndrewMultilingual=gratitude/neutral)

  Tier 2: Nuclear Edge TTS params
          (rate up to +55%, pitch down to -15Hz -- far beyond previous timid values)

  Tier 3: Post-RVC pitch envelope + time-stretch
          (operates on OUTPUT waveform, bypasses HuBERT washout entirely)

Pipeline per WAV:
  Edge TTS (voice + pitch + rate)
    -> RVC v2 (rmvpe, dynamic semitone, FAISS IVF256)
    -> Silence Gate (sigmoid, input-guided)
    -> Post-Process (pitch shift + time stretch on RVC output)
    -> Analog Warmth Saturation
    -> WAV file
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

NEUTRAL_PARAMS = {
    "voice":               "en-US-AndrewMultilingualNeural",
    "edge_pitch_hz":       0,
    "edge_rate_pct":       0,
    "rvc_pitch_semitones": 2,
    "index_rate":          0.88,
    "rms_mix_rate":        0.20,
    "protect":             0.50,
    "post_pitch_semitones": 0,
    "post_time_stretch":   1.0,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# TIER 1 + 2 -- Edge TTS with voice + nuclear params
# ─────────────────────────────────────────────────────────────

async def _tts_attempt(text: str, voice: str, pitch_str: str, rate_str: str) -> bytes:
    comm = edge_tts.Communicate(text=text, voice=voice, pitch=pitch_str, rate=rate_str)
    buf  = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    data = buf.getvalue()
    if len(data) < 200:
        raise RuntimeError("Edge TTS returned empty payload")
    return data


async def generate_tts(text: str, voice: str, pitch_hz: int, rate_pct: int) -> bytes:
    pitch_str = f"{pitch_hz:+d}Hz"
    rate_str  = f"{rate_pct:+d}%"
    print(f"     [TTS] voice={voice}  pitch={pitch_str}  rate={rate_str}")
    for attempt in range(1, 6):
        try:
            return await _tts_attempt(text, voice, pitch_str, rate_str)
        except Exception as exc:
            wait = 2 ** attempt
            if attempt < 5:
                print(f"     [TTS] attempt {attempt} failed ({type(exc).__name__}), retry in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Edge TTS failed after 5 attempts: {exc}") from exc


# ─────────────────────────────────────────────────────────────
# RVC v2 INFERENCE
# ─────────────────────────────────────────────────────────────

def run_rvc(raw_bytes: bytes, pitch: int, index_rate: float,
            rms_mix: float, protect: float) -> bytes:
    """RVC v2 subprocess call -- rmvpe F0, FAISS IVF256 index."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
        in_path  = fin.name
        out_path = fout.name
        fin.write(raw_bytes)

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        py  = PYTHON_VENV if os.path.exists(PYTHON_VENV) else sys.executable
        cmd = [
            py, "-m", "infer.cli",
            "--model",       SUBARU_MODEL,
            "--input",       in_path,
            "--output",      out_path,
            "--f0-method",   "rmvpe",
            "--pitch",       str(pitch),
            "--index-rate",  str(index_rate),
            "--rms-mix-rate", str(rms_mix),
            "--protect",     str(protect),
            "--overwrite",
        ]
        if os.path.exists(SUBARU_INDEX):
            cmd.extend(["--index", SUBARU_INDEX])

        print(f"     [RVC] pitch={pitch:+d}st  index_rate={index_rate}  rms_mix={rms_mix}")
        res = subprocess.run(cmd, cwd=RVC_ENGINE_DIR,
                             capture_output=True, text=True,
                             encoding="utf-8", errors="ignore", env=env)

        if res.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 500:
            err = (res.stderr or res.stdout or "").strip()[:200]
            raise RuntimeError(f"RVC failed (code {res.returncode}): {err}")

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            try: os.remove(p)
            except Exception: pass


# ─────────────────────────────────────────────────────────────
# SILENCE GATE
# ─────────────────────────────────────────────────────────────

def apply_silence_gate(tts_bytes: bytes, rvc_bytes: bytes) -> bytes:
    """
    Input-guided sigmoid gate.
    Source TTS energy controls muting of RVC output during silent frames.
    Prevents vocoder 226 Hz drone in pauses.
    """
    y_src,  sr_src  = sf.read(io.BytesIO(tts_bytes), dtype="float32")
    y_conv, sr_conv = sf.read(io.BytesIO(rvc_bytes),  dtype="float32")

    if y_src.ndim  > 1: y_src  = np.mean(y_src,  axis=1)
    if y_conv.ndim > 1: y_conv = np.mean(y_conv, axis=1)

    if sr_src != sr_conv:
        y_src = librosa.resample(y_src, orig_sr=sr_src, target_sr=sr_conv)

    n        = min(len(y_src), len(y_conv))
    y_src    = y_src[:n]
    y_conv   = y_conv[:n]

    win      = int(0.025 * sr_conv)
    hop      = int(0.010 * sr_conv)
    rms      = np.zeros(n, dtype=np.float32)
    for i in range(0, n - win, hop):
        rms[i:i + hop] = np.sqrt(np.mean(y_src[i:i + win] ** 2))

    sigma    = max(1, int(0.025 * sr_conv / hop))
    rms_s    = gaussian_filter1d(rms, sigma=sigma)
    gate     = 1.0 / (1.0 + np.exp(-(rms_s - 0.0015) / 0.0003))

    out = io.BytesIO()
    sf.write(out, y_conv * gate, sr_conv, format="WAV")
    return out.getvalue()


# ─────────────────────────────────────────────────────────────
# TIER 3 -- Post-RVC Pitch + Time-Stretch Envelope
# ─────────────────────────────────────────────────────────────

def apply_post_rvc(audio_bytes: bytes,
                   pitch_semitones: int,
                   time_stretch: float) -> bytes:
    """
    Apply pitch shift and time-stretch to the RVC OUTPUT waveform.
    Bypasses HuBERT tokenization -- operates directly on the reconstructed signal.
    
    pitch_semitones: positive = higher, negative = lower
    time_stretch:    <1.0 = slower (grief), >1.0 = faster (not used here)
    """
    if pitch_semitones == 0 and abs(time_stretch - 1.0) < 0.01:
        return audio_bytes  # nothing to do

    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Time-stretch first (affects duration, not pitch)
    if abs(time_stretch - 1.0) >= 0.01:
        print(f"     [POST] time_stretch={time_stretch:.2f}")
        y = librosa.effects.time_stretch(y, rate=time_stretch)

    # Then pitch shift (affects pitch, not duration)
    if pitch_semitones != 0:
        print(f"     [POST] pitch_shift={pitch_semitones:+d} semitones")
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_semitones)

    # Re-normalize to 0.92 peak after processing
    peak = np.max(np.abs(y))
    if peak > 0.001:
        y = y / peak * 0.92

    out = io.BytesIO()
    sf.write(out, y, sr, format="WAV")
    return out.getvalue()


# ─────────────────────────────────────────────────────────────
# ANALOG WARMTH SATURATION
# ─────────────────────────────────────────────────────────────

def apply_warmth(audio_bytes: bytes) -> bytes:
    """
    Soft-clip tanh saturation -- adds chest resonance body.
    Mimics tape/tube warmth: slightly boosts low-mids, gentle harmonic distortion.
    """
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    peak = np.max(np.abs(y))
    if peak > 0.001:
        y = y / peak * 0.95

    # Drive -> tanh -> normalize: adds harmonic richness without clipping
    y = np.tanh(y * 1.10) / 1.05

    out = io.BytesIO()
    sf.write(out, y, sr, format="WAV")
    return out.getvalue()


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(text: str, p: dict, label: str, out_path: str) -> None:
    """
    Execute full 5-stage pipeline for one WAV.
    Skips if output already exists and is valid (resume support).
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"  [OK] {label:10s} already exists -- skipping")
        return

    print(f"  -> {label:10s} generating...")

    # Stage 1: TTS
    tts_bytes = asyncio.run(generate_tts(
        text     = text,
        voice    = p["voice"],
        pitch_hz = p["edge_pitch_hz"],
        rate_pct = p["edge_rate_pct"],
    ))

    # Stage 2: RVC
    rvc_bytes = run_rvc(
        raw_bytes  = tts_bytes,
        pitch      = p["rvc_pitch_semitones"],
        index_rate = p["index_rate"],
        rms_mix    = p["rms_mix_rate"],
        protect    = p["protect"],
    )

    # Stage 3: Silence Gate
    gated_bytes = apply_silence_gate(tts_bytes, rvc_bytes)

    # Stage 4: Post-RVC pitch + time-stretch (Tier 3)
    post_bytes = apply_post_rvc(
        audio_bytes     = gated_bytes,
        pitch_semitones = p.get("post_pitch_semitones", 0),
        time_stretch    = p.get("post_time_stretch",    1.0),
    )

    # Stage 5: Analog warmth
    final_bytes = apply_warmth(post_bytes)

    with open(out_path, "wb") as f:
        f.write(final_bytes)

    size_kb = len(final_bytes) // 1024
    print(f"  [OK] {label:10s} -> {os.path.basename(out_path)}  ({size_kb} KB)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    with open(PARAMS_FILE, encoding="utf-8") as f:
        ALL_PARAMS = json.load(f)

    print("=" * 68)
    print("  SUBARU EMOTION EXPERIMENT -- Phase 1 v2 (Best Accuracy)")
    print("  Tier 1: Per-emotion voice  |  Tier 2: Nuclear params")
    print("  Tier 3: Post-RVC pitch envelope + time-stretch")
    print("=" * 68)

    generated = []

    for idx, emotion, text in TEST_LINES:
        flat_path  = os.path.join(OUTPUT_DIR, f"subaru_exp_{idx:02d}_{emotion}_FLAT.wav")
        emo_path   = os.path.join(OUTPUT_DIR, f"subaru_exp_{idx:02d}_{emotion}_EMOTION.wav")
        emo_params = ALL_PARAMS.get(emotion, ALL_PARAMS["neutral"])

        print(f"\n[{idx}/5] {emotion.upper()}")
        print(f'  "{text}"')

        run_pipeline(text, NEUTRAL_PARAMS, "FLAT",    flat_path)
        run_pipeline(text, emo_params,     "EMOTION", emo_path)

        generated += [flat_path, emo_path]

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    ok = [p for p in generated if os.path.exists(p) and os.path.getsize(p) > 1000]
    print(f"  DONE: {len(ok)}/10 WAVs generated")
    print("=" * 68)
    for p in ok:
        print(f"  {os.path.basename(p):55s} {os.path.getsize(p)//1024:>5} KB")

    # ── Copy to artifacts ────────────────────────────────────
    print("\n[Copying to artifacts...]")
    import shutil
    copied = 0
    for p in ok:
        dest = os.path.join(ARTIFACT_DIR, os.path.basename(p))
        try:
            shutil.copy2(p, dest)
            copied += 1
        except Exception as e:
            print(f"  [WARN] {os.path.basename(p)}: {e}")
    print(f"[OK] {copied}/{len(ok)} WAVs copied to artifacts")

    return len(ok)


if __name__ == "__main__":
    n = main()
    sys.exit(0 if n == 10 else 1)

