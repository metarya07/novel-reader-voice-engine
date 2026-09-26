#!/usr/bin/env python3
"""
Interactive CLI rating tool for Subaru Natsuki voice experiment.
Evaluates FLAT vs EMOTION audio for each emotion pair and logs ratings.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
SAMPLES_DIR = BASE_DIR / "output"
PARAMS_FILE = REPO_ROOT / "params" / "subaru_params.json"
RATINGS_LOG_FILE = BASE_DIR / "ratings_log.jsonl"

TEST_LINES = [
    ("rage", "I'll save you, I swear it! No matter how many times it takes!"),
    ("grief", "I'm sorry. I'm so sorry, Rem. I'm nothing. I'm worthless."),
    ("comedic", "Wait — seriously?! That actually worked?! Ha! I'm a genius!"),
    ("cold_threat", "Don't come near her. If you take one more step, I will end you."),
    ("gratitude", "Emilia… thank you. Just — thank you for believing in me."),
]


def prompt_rating(prompt_text: str) -> int:
    """Prompt user for an integer rating between 1 and 5 with validation."""
    while True:
        try:
            val = input(prompt_text).strip()
            if val.lower() in ("q", "quit", "exit"):
                print("\nRating session terminated by user.")
                sys.exit(0)
            score = int(val)
            if 1 <= score <= 5:
                return score
            print("  [!] Please enter a whole number between 1 and 5 (or 'q' to quit).")
        except ValueError:
            print("  [!] Invalid input. Please enter an integer from 1 to 5.")
        except (KeyboardInterrupt, EOFError):
            print("\nRating session cancelled.")
            sys.exit(0)


def prompt_notes(prompt_text: str) -> str:
    """Prompt user for optional notes."""
    try:
        return input(prompt_text).strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def main():
    print("================================================================================")
    print("Subaru Natsuki Voice Experiment — Audio Evaluation Tool")
    print("================================================================================")
    print("Listen to both versions (FLAT vs EMOTION) using the clickable links below,")
    print("then rate each from 1 (poor) to 5 (excellent).\n")

    if not PARAMS_FILE.exists():
        print(f"[Error] Params file not found at: {PARAMS_FILE}")
        sys.exit(1)

    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        params = json.load(f)

    for i, (emotion, line) in enumerate(TEST_LINES, start=1):
        idx_str = f"{i:02d}"
        flat_path = SAMPLES_DIR / f"subaru_exp_{idx_str}_{emotion}_FLAT.wav"
        emotion_path = SAMPLES_DIR / f"subaru_exp_{idx_str}_{emotion}_EMOTION.wav"

        flat_uri = f"file:///{flat_path.as_posix()}"
        emotion_uri = f"file:///{emotion_path.as_posix()}"

        print("--------------------------------------------------------------------------------")
        print(f"[{idx_str}/05] Emotion: {emotion.upper()}")
        print(f"Line: \"{line}\"")
        print(f"FLAT WAV:    {flat_uri}")
        print(f"EMOTION WAV: {emotion_uri}")
        print("--------------------------------------------------------------------------------")

        if not flat_path.exists():
            print(f"[Warning] FLAT file not found on disk: {flat_path}")
        if not emotion_path.exists():
            print(f"[Warning] EMOTION file not found on disk: {emotion_path}")

        flat_rating = prompt_rating("Rate FLAT (1-5): ")
        emotion_rating = prompt_rating("Rate EMOTION (1-5): ")
        notes = prompt_notes("Notes (optional): ")

        emotion_params = params.get(emotion, params.get("neutral", {}))
        # Keep relevant tuning parameters
        params_used = {
            "edge_pitch_hz": emotion_params.get("edge_pitch_hz", 0),
            "edge_rate_pct": emotion_params.get("edge_rate_pct", 0),
            "rvc_pitch_semitones": emotion_params.get("rvc_pitch_semitones", 2),
            "index_rate": emotion_params.get("index_rate", 0.88),
            "rms_mix_rate": emotion_params.get("rms_mix_rate", 0.20),
            "protect": emotion_params.get("protect", 0.50),
        }

        entry = {
            "emotion": emotion,
            "flat_rating": flat_rating,
            "emotion_rating": emotion_rating,
            "params_used": params_used,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(RATINGS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"[Saved] Logged rating for {emotion.upper()} (FLAT: {flat_rating}, EMOTION: {emotion_rating})\n")

    print("================================================================================")
    print(f"All ratings saved to: {RATINGS_LOG_FILE}")
    print("You can now run optimize_params.py to tune the parameter weights.")
    print("================================================================================")


if __name__ == "__main__":
    main()
