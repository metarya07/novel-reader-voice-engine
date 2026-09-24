# voice-model-server

Python gRPC server — all ML models, GPU inference, emotion classification.

## Responsibilities

- Serve `VoiceService` gRPC endpoints (port 50051)
- Run emotion classification on CPU (DistilRoBERTa)
- Run Edge TTS with emotion-tuned prosody params
- Run RVC v2 inference on RTX 4050 CUDA
- Apply silence gate to remove vocoder drone
- Return typed `StageResult` for every pipeline stage

## Structure

```
voice-model-server/
├── grpc_server.py              # gRPC servicer + server bootstrap
├── services/
│   ├── emotion_classifier.py   # DistilRoBERTa + keyword fallback
│   ├── param_store.py          # loads params/*.json per character
│   ├── tts_service.py          # Edge TTS wrapper
│   ├── rvc_service.py          # RVC v2 CUDA inference
│   └── silence_gate.py         # numpy sigmoid gate
├── exception/
│   ├── result.py               # StageResult dataclass
│   ├── error_codes.py          # CUDA_OOM, TIMEOUT, MODEL_NOT_FOUND etc
│   └── fallback_chain.py       # per-stage fallback logic
├── params/
│   ├── subaru_params.json      # emotion → prosody params for Subaru
│   ├── emilia_params.json      # emotion → prosody params for Emilia
│   └── default_params.json     # safe fallback for unknown characters
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/voice_service.proto
python grpc_server.py
```

## Environment

- Python 3.11
- CUDA 12.4
- PyTorch 2.x
- RTX 4050 Laptop GPU (6GB VRAM)
