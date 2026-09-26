#!/usr/bin/env python3
"""
Parameter optimization script for Subaru Natsuki voice profile.
Reads ratings_log.jsonl, fits surrogate model for each rated emotion,
and uses scipy.optimize.minimize to find the optimal parameter set.
Updates params/subaru_params.json and displays a before/after comparison table.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.optimize import minimize

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
PARAMS_FILE = REPO_ROOT / "params" / "subaru_params.json"
RATINGS_LOG_FILE = BASE_DIR / "ratings_log.jsonl"

# Parameter scales and bounds: [pitch_hz, rate_pct, rvc_pitch_semitones, index_rate]
BOUNDS = [
    (-20.0, 20.0),   # edge_pitch_hz
    (-50.0, 50.0),   # edge_rate_pct
    (-12.0, 12.0),   # rvc_pitch_semitones
    (0.10, 1.00),    # index_rate
]

SCALES = np.array([10.0, 25.0, 3.0, 0.25], dtype=np.float64)


def load_ratings() -> Dict[str, List[Dict[str, Any]]]:
    """Load ratings from JSONL file and group by emotion."""
    if not RATINGS_LOG_FILE.exists():
        return {}

    emotions_data: Dict[str, List[Dict[str, Any]]] = {}
    with open(RATINGS_LOG_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                emotion = record.get("emotion")
                if emotion:
                    emotions_data.setdefault(emotion, []).append(record)
            except json.JSONDecodeError as err:
                print(f"[Warning] Skipping malformed line {line_no} in ratings log: {err}")

    return emotions_data


def extract_vector(params: Dict[str, Any]) -> np.ndarray:
    """Extract [pitch_hz, rate_pct, rvc_pitch, index_rate] vector."""
    return np.array([
        float(params.get("edge_pitch_hz", 0)),
        float(params.get("edge_rate_pct", 0)),
        float(params.get("rvc_pitch_semitones", 2)),
        float(params.get("index_rate", 0.88)),
    ], dtype=np.float64)


def optimize_emotion_params(
    emotion: str,
    records: List[Dict[str, Any]],
    current_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """
    Use scipy.optimize.minimize on surrogate response surface to maximize emotion rating.
    Returns: (optimized_params_dict, avg_emotion_rating, avg_flat_rating)
    """
    X_list = []
    y_emo_list = []
    y_flat_list = []

    for r in records:
        pu = r.get("params_used", {})
        x = extract_vector(pu)
        X_list.append(x)
        y_emo_list.append(float(r.get("emotion_rating", 3)))
        y_flat_list.append(float(r.get("flat_rating", 3)))

    X = np.array(X_list)
    y_emo = np.array(y_emo_list)
    y_flat = np.array(y_flat_list)

    avg_emotion_rating = float(np.mean(y_emo))
    avg_flat_rating = float(np.mean(y_flat))

    best_idx = int(np.argmax(y_emo))
    x_best = X[best_idx].copy()
    current_vec = extract_vector(current_params)
    x0 = current_vec.copy()

    # Surrogate kernel-weighted rating objective
    def surrogate_loss(x: np.ndarray) -> float:
        # Normalized distance to evaluated sample points
        diffs = (X - x) / SCALES
        sq_dist = np.sum(diffs ** 2, axis=1)
        weights = np.exp(-0.5 * sq_dist) + 1e-6
        pred_rating = np.sum(weights * y_emo) / np.sum(weights)

        # Gradient bonus: if emotion rating < flat rating, pull towards neutral baseline
        # If emotion rating > flat rating, give positive reinforcement around high-rated regions
        advantage = np.mean(y_emo - y_flat)
        reg_target = x_best if advantage >= 0 else np.array([0.0, 2.0, 2.0, 0.88])
        reg_penalty = 0.05 * np.sum(((x - reg_target) / SCALES) ** 2)

        # We minimize negative predicted rating + regularization
        return float(-pred_rating + reg_penalty)

    res = minimize(
        surrogate_loss,
        x0=x0,
        method="L-BFGS-B",
        bounds=BOUNDS
    )

    x_opt = res.x if res.success else x_best

    opt_dict = {
        "voice": current_params.get("voice", "en-US-AndrewMultilingualNeural"),
        "edge_pitch_hz": int(round(x_opt[0])),
        "edge_rate_pct": int(round(x_opt[1])),
        "rvc_pitch_semitones": int(round(x_opt[2])),
        "index_rate": round(float(np.clip(x_opt[3], 0.10, 1.00)), 2),
        "rms_mix_rate": current_params.get("rms_mix_rate", 0.20),
        "protect": current_params.get("protect", 0.50),
        "notes": current_params.get("notes", "")
    }

    return opt_dict, avg_emotion_rating, avg_flat_rating


def print_comparison_table(
    before_params: Dict[str, Any],
    after_params: Dict[str, Any],
    stats: Dict[str, Tuple[float, float, int]]
):
    """Print markdown and ASCII before/after comparison table."""
    print("\n" + "=" * 86)
    print("PARAM TUNING SUMMARY: BEFORE vs AFTER OPTIMIZATION")
    print("=" * 86)
    header = f"| {'Emotion':<12} | {'Parameter':<20} | {'Before':<8} | {'After':<8} | {'Delta':<8} | {'Avg Rating (E/F)':<16} |"
    sep = f"|{'-'*14}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*18}|"
    print(header)
    print(sep)

    tracked_keys = ["edge_pitch_hz", "edge_rate_pct", "rvc_pitch_semitones", "index_rate"]

    for emotion, new_cfg in after_params.items():
        if emotion not in stats:
            continue
        old_cfg = before_params.get(emotion, {})
        avg_e, avg_f, count = stats[emotion]
        rating_str = f"{avg_e:.1f} / {avg_f:.1f} (n={count})"

        for idx, key in enumerate(tracked_keys):
            old_val = old_cfg.get(key, "-")
            new_val = new_cfg.get(key, "-")

            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                delta_val = new_val - old_val
                delta_str = f"{delta_val:+.2f}" if isinstance(new_val, float) else f"{delta_val:+d}"
            else:
                delta_str = "-"

            emo_col = emotion if idx == 0 else ""
            stat_col = rating_str if idx == 0 else ""
            print(f"| {emo_col:<12} | {key:<20} | {str(old_val):<8} | {str(new_val):<8} | {delta_str:<8} | {stat_col:<16} |")
        print(f"|{'-'*14}|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*18}|")


def main():
    print("================================================================================")
    print("Subaru Parameter Optimizer (scipy.optimize.minimize)")
    print("================================================================================")

    if not RATINGS_LOG_FILE.exists():
        print(f"[!] No ratings found at: {RATINGS_LOG_FILE}")
        print("    Please run 'rate_wavs.py' first to evaluate and log ratings.")
        sys.exit(0)

    ratings_by_emotion = load_ratings()
    if not ratings_by_emotion:
        print(f"[!] Ratings file {RATINGS_LOG_FILE} is empty.")
        print("    Please run 'rate_wavs.py' first to log audio ratings.")
        sys.exit(0)

    if not PARAMS_FILE.exists():
        print(f"[Error] Params file not found at: {PARAMS_FILE}")
        sys.exit(1)

    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        original_params = json.load(f)

    updated_params = dict(original_params)
    stats: Dict[str, Tuple[float, float, int]] = {}

    print(f"\nFound logged ratings for emotions: {list(ratings_by_emotion.keys())}")

    for emotion, records in ratings_by_emotion.items():
        curr_cfg = original_params.get(emotion, original_params.get("neutral", {}))
        opt_cfg, avg_emo, avg_flat = optimize_emotion_params(emotion, records, curr_cfg)
        updated_params[emotion] = opt_cfg
        stats[emotion] = (avg_emo, avg_flat, len(records))
        print(f" - Optimized {emotion:<12}: Avg Emotion Rating = {avg_emo:.2f}, Flat = {avg_flat:.2f} ({len(records)} entries)")

    # Save updated parameters
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_params, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n[Saved] Updated {PARAMS_FILE}")
    print_comparison_table(original_params, updated_params, stats)


if __name__ == "__main__":
    main()
