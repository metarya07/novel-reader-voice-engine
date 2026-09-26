#!/usr/bin/env python3
"""
Subaru Natsuki -- Emotion Extraction & Dedicated FAISS Index Builder
===================================================================
1. Analyzes all 3,733 sliced Subaru clips and pre-computed RMVPE F0 tracks
2. Extracts acoustic properties (RMS energy, F0 mean/max/std, spectral dynamics)
3. Clusters clips into 5 distinct emotional acting profiles:
   - RAGE: High RMS energy + high F0 peaks (battle screams, furious determination)
   - GRIEF: Low RMS energy + low F0 register (crying, sobbing, vulnerable breakdown)
   - COMEDIC: High F0 variance + dynamic inflections (cheeky anime banter, boasting)
   - COLD_THREAT: Low F0 mean + ultra-flat variance (deadly calm, chilling baritone)
   - NEUTRAL: Balanced conversational dialogue
4. Concatenates matching HuBERT 768-dim embeddings from rvc-engine/logs/subaru/3_feature768/
5. Builds dedicated FAISS IVF indices for each emotion:
   - subaru_rage_v2.index
   - subaru_grief_v2.index
   - subaru_comedic_v2.index
   - subaru_cold_threat_v2.index
   - subaru_neutral_v2.index
6. Computes empirical pitch & pacing statistics per emotion for automated TTS calibration.
"""

import os
import sys
import glob
import time
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import faiss

# Absolute Paths
BASE_DIR      = r"a:\Projects\novel reader\rvc-engine\logs\subaru"
WAV_DIR       = os.path.join(BASE_DIR, "0_gt_wavs")
F0_DIR        = os.path.join(BASE_DIR, "2a_f0")
FEA_DIR       = os.path.join(BASE_DIR, "3_feature768")
OUTPUT_DIR    = r"a:\Projects\novel-reader-voice-engine\models\subaru\indices"
STATS_OUT     = r"a:\Projects\novel-reader-voice-engine\params\subaru_empirical_stats.json"
VOICE_SERVER_INDEX_DIR = r"a:\Projects\novel reader\voice-server\models\subaru"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_SERVER_INDEX_DIR, exist_ok=True)


def analyze_clips():
    print("================================================================================")
    print("  SUBARU NATSUKI -- EMOTION EXTRACTION & FAISS RETRIEVAL BUILDER")
    print("  Target: 3,733 Pre-computed Sean Chiplock Segments")
    print("  Engine: HuBERT 768-dim + RMVPE Pitch Contours")
    print("================================================================================\n")

    wav_files = glob.glob(os.path.join(WAV_DIR, "*.wav"))
    total_files = len(wav_files)
    print(f"[*] Found {total_files} audio segments in {WAV_DIR}")
    print("[*] Scanning acoustic properties and matching HuBERT feature tensors...\n")

    records = []
    t0 = time.time()

    for idx, wpath in enumerate(wav_files, start=1):
        stem = Path(wpath).stem
        f0_path = os.path.join(F0_DIR, f"{stem}.wav.npy")
        fea_path = os.path.join(FEA_DIR, f"{stem}.npy")

        if not os.path.exists(f0_path) or not os.path.exists(fea_path):
            continue

        try:
            # Audio energy
            audio, sr = sf.read(wpath, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dur = len(audio) / sr

            # RMVPE F0 pitch
            f0 = np.load(f0_path)
            voiced = f0[f0 > 10.0]

            if len(voiced) < 5:
                continue

            f0_mean = float(np.mean(voiced))
            f0_max = float(np.max(voiced))
            f0_min = float(np.min(voiced))
            f0_std = float(np.std(voiced))

            records.append({
                "stem": stem,
                "wpath": wpath,
                "fea_path": fea_path,
                "rms": rms,
                "duration": dur,
                "f0_mean": f0_mean,
                "f0_max": f0_max,
                "f0_min": f0_min,
                "f0_std": f0_std,
            })
        except Exception as e:
            continue

        if idx % 500 == 0 or idx == total_files:
            elapsed = time.time() - t0
            print(f"    Processed {idx}/{total_files} segments ({len(records)} valid) [{elapsed:.1f}s]")

    print(f"\n[OK] Extracted features for {len(records)} active speech segments.\n")
    return records


def cluster_emotions(records):
    print("[*] Computing acoustic thresholds for emotional separation...")
    all_rms = np.array([r["rms"] for r in records])
    all_f0 = np.array([r["f0_mean"] for r in records])
    all_f0_std = np.array([r["f0_std"] for r in records])
    all_f0_max = np.array([r["f0_max"] for r in records])

    rms_p75 = float(np.percentile(all_rms, 75))
    rms_p35 = float(np.percentile(all_rms, 35))
    f0_p75  = float(np.percentile(all_f0, 75))
    f0_p30  = float(np.percentile(all_f0, 30))
    std_p75 = float(np.percentile(all_f0_std, 75))
    std_p30 = float(np.percentile(all_f0_std, 30))
    max_p80 = float(np.percentile(all_f0_max, 80))

    print(f"    * RMS 75th percentile (Loud/Intensity):    {rms_p75:.4f}")
    print(f"    * RMS 35th percentile (Soft/Intimate):     {rms_p35:.4f}")
    print(f"    * F0 Mean 75th percentile (High Register): {f0_p75:.1f} Hz")
    print(f"    * F0 Mean 30th percentile (Deep Chest):    {f0_p30:.1f} Hz")
    print(f"    * F0 Max 80th percentile (Shout Peaks):    {max_p80:.1f} Hz")
    print(f"    * F0 Std 75th percentile (Animated):       {std_p75:.1f} Hz")
    print(f"    * F0 Std 30th percentile (Monotone/Cold):  {std_p30:.1f} Hz\n")

    clusters = {
        "rage": [],
        "grief": [],
        "comedic": [],
        "cold_threat": [],
        "neutral": [],
    }

    for r in records:
        # 1. RAGE: High energy + screaming pitch peaks
        if (r["rms"] >= rms_p75 and r["f0_max"] >= max_p80) or (r["f0_mean"] >= f0_p75 and r["rms"] >= rms_p75):
            clusters["rage"].append(r)
        # 2. GRIEF: Soft energy + low chest breakdown + sorrowful pitch
        elif r["rms"] <= rms_p35 and r["f0_mean"] <= f0_p30:
            clusters["grief"].append(r)
        # 3. COMEDIC: High pitch variance (wide bouncing inflections) + medium-high energy
        elif r["f0_std"] >= std_p75 and r["rms"] >= rms_p35:
            clusters["comedic"].append(r)
        # 4. COLD THREAT: Low chest register + ultra-low variance (deadly flat, monotone)
        elif r["f0_mean"] <= f0_p30 and r["f0_std"] <= std_p30:
            clusters["cold_threat"].append(r)
        # 5. NEUTRAL: Natural conversational balance
        else:
            clusters["neutral"].append(r)

    print("================================================================================")
    print("  EMOTION CLUSTER DISTRIBUTION")
    print("================================================================================")
    stats_data = {}
    for emo, items in clusters.items():
        if items:
            avg_f0 = np.mean([x["f0_mean"] for x in items])
            avg_rms = np.mean([x["rms"] for x in items])
            avg_std = np.mean([x["f0_std"] for x in items])
            print(f"  * {emo.upper():12s}: {len(items):>5} segments | F0 Mean: {avg_f0:>5.1f} Hz | Std: {avg_std:>4.1f} Hz | RMS: {avg_rms:.4f}")
            stats_data[emo] = {
                "segment_count": len(items),
                "f0_mean_hz": round(float(avg_f0), 1),
                "f0_std_hz": round(float(avg_std), 1),
                "rms_energy": round(float(avg_rms), 5),
            }

    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\n[OK] Saved empirical prosody statistics -> {STATS_OUT}\n")

    return clusters


def build_emotion_index(emo_name, records):
    print(f"[*] Building FAISS IVF Index for emotion: [{emo_name.upper()}] ({len(records)} segments)...")
    if not records:
        print(f"    [!] No records for {emo_name}, skipping.")
        return

    features = []
    for r in records:
        try:
            fea = np.load(r["fea_path"])
            if fea.ndim == 2 and fea.shape[1] == 768:
                features.append(fea)
        except Exception:
            continue

    if not features:
        print(f"    [!] Failed to load features for {emo_name}.")
        return

    big_fea = np.concatenate(features, axis=0)
    num_vectors = big_fea.shape[0]
    print(f"    Total HuBERT 768-dim vectors: {num_vectors:,}")

    # Random shuffle
    np.random.seed(42)
    big_fea = big_fea[np.random.permutation(num_vectors)]

    # Downsample if cluster is very large
    if num_vectors > 150000:
        print(f"    Clustering {num_vectors:,} vectors into 10,000 centers with MiniBatchKMeans...")
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=10000,
            batch_size=2048,
            init="random",
            compute_labels=False,
            random_state=42
        )
        big_fea = kmeans.fit(big_fea).cluster_centers_
        num_vectors = big_fea.shape[0]
        print(f"    Compressed to {num_vectors:,} cluster centers.")

    # Calculate optimal IVF clusters
    n_ivf = max(1, min(int(16 * np.sqrt(num_vectors)), num_vectors // 39, 256))
    print(f"    Training FAISS IVF{n_ivf},Flat index...")

    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    index.train(big_fea)
    index.add(big_fea)

    out_file = f"subaru_{emo_name}_v2.index"
    save_path = os.path.join(OUTPUT_DIR, out_file)
    faiss.write_index(index, save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"    [SUCCESS] Saved: {save_path} ({file_size_mb:.2f} MB)")

    # Also deploy to voice-server
    server_path = os.path.join(VOICE_SERVER_INDEX_DIR, out_file)
    faiss.write_index(index, server_path)
    print(f"    [DEPLOYED] Synced to voice-server: {server_path}\n")


def main():
    records = analyze_clips()
    clusters = cluster_emotions(records)

    print("================================================================================")
    print("  GENERATING DEDICATED EMOTIONAL FAISS INDICES")
    print("================================================================================")

    for emo_name, items in clusters.items():
        build_emotion_index(emo_name, items)

    print("================================================================================")
    print("  ALL EMOTION INDICES BUILT & DEPLOYED SUCCESSFULLY!")
    print("================================================================================")
    print(f"Target Directory: {OUTPUT_DIR}")
    for idx_file in glob.glob(os.path.join(OUTPUT_DIR, "*.index")):
        sz = os.path.getsize(idx_file) / (1024 * 1024)
        print(f"  * {os.path.basename(idx_file):30s} [{sz:.2f} MB]")


if __name__ == "__main__":
    main()
