# experiments

Standalone Python scripts for tuning emotion-aware prosody parameters per character.
No server changes, no classifier — pure param tuning with manual labels.

## Purpose

Tune the `params/*.json` files by:
1. Running the experiment script for a character
2. Listening to the generated WAVs
3. Rating each WAV
4. Running the optimizer to find best params

## Structure

```
experiments/
├── subaru/
│   ├── run_experiment.py        # generates flat vs emotion WAV pairs
│   ├── optimize_params.py       # scipy minimize on your ratings
│   └── ratings_log.jsonl        # your 1-5 ratings per WAV
├── emilia/
│   ├── run_experiment.py
│   ├── optimize_params.py
│   └── ratings_log.jsonl
├── test_lines/
│   ├── subaru_test_lines.json   # 5 emotion-labeled lines per character
│   └── emilia_test_lines.json
└── README.md
```

## Subaru Test Lines

| # | Emotion | Line |
|---|---------|------|
| 1 | `rage` | "I'll save you, I swear it! No matter how many times it takes!" |
| 2 | `grief` | "I'm sorry. I'm so sorry, Rem. I'm nothing. I'm worthless." |
| 3 | `comedic` | "Wait — seriously?! That actually worked?! Ha! I'm a genius!" |
| 4 | `cold_threat` | "Don't come near her. If you take one more step, I will end you." |
| 5 | `gratitude` | "Emilia… thank you. Just — thank you for believing in me." |

## Output Files

```
samples/
  subaru_exp_01_rage_FLAT.wav
  subaru_exp_01_rage_EMOTION.wav
  subaru_exp_02_grief_FLAT.wav
  subaru_exp_02_grief_EMOTION.wav
  ...
```

## Usage

```bash
# Run experiment
python experiments/subaru/run_experiment.py

# Rate WAVs (opens each pair, you score 1-5)
python experiments/subaru/rate_wavs.py

# Optimize params from your ratings
python experiments/subaru/optimize_params.py
# → writes params/subaru_params.json
```
