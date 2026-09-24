# 🎙️ Novel Reader Voice Engine

> **Hybrid Java + Python emotion-aware RVC v2 voice pipeline**
> Built for Re:Zero English Dub characters — Subaru (Sean Chiplock) & Emilia (Kayli Mills)

---

## Architecture Overview

```
Frontend (Novel Reader)
        │  HTTP / WebSocket
        ▼
┌──────────────────────────────────┐
│  voice-orchestrator (Java 21)    │  Pipeline, cache, API, queues
│  Spring Boot 3 + Project Reactor │
└──────────────┬───────────────────┘
               │  gRPC (localhost:50051)
               │  Protocol Buffers
               ▼
┌──────────────────────────────────┐
│  voice-model-server (Python)     │  All ML models + GPU inference
│  FastAPI + gRPC Server           │
│                                  │
│  ┌─────────────────────────────┐ │
│  │ Emotion Classifier (CPU)    │ │  DistilRoBERTa
│  │ Param Store                 │ │  Per-character JSON
│  │ Edge TTS                    │ │  Microsoft Neural
│  │ RVC v2 (RTX 4050 CUDA)      │ │  HuBERT + RMVPE + HifiGAN
│  │ Silence Gate                │ │  Sigmoid numpy gate
│  └─────────────────────────────┘ │
└──────────────────────────────────┘
```

## Modules

| Module | Language | Status | Description |
|--------|----------|--------|-------------|
| `voice-orchestrator` | Java 21 | 🔲 Planned | Spring Boot pipeline, gRPC client, cache |
| `voice-model-server` | Python 3.11 | 🔲 Planned | gRPC server, RVC v2, emotion classifier |
| `proto` | Protobuf | 🔲 Planned | Shared contract between Java ↔ Python |
| `experiments` | Python | 🔲 Planned | Subaru emotion param tuning scripts |
| `params` | JSON | 🔲 Planned | Per-character emotion→prosody tables |

## Characters

| Character | Voice Actor | Model | Status |
|-----------|-------------|-------|--------|
| Subaru Natsuki | Sean Chiplock | `subaru_e50_s22500.pth` | ✅ Trained |
| Emilia | Kayli Mills | `emilia_e50.pth` | 🔄 Training |
| Rem | Brianna Knickerbocker | — | 🔲 Planned |
| Beatrice | Kira Buckland | — | 🔲 Planned |

## Emotion Pipeline

```
Text → [Emotion Classifier] → [Param Store] → [Edge TTS SSML] → [RVC v2] → [Silence Gate] → Audio
```

Supported emotions per character:
- `neutral` `rage` `grief` `fear` `joy` `surprise` `gratitude` `cold_threat`

## Hardware

- **GPU**: NVIDIA GeForce RTX 4050 Laptop (6GB VRAM, CUDA 12.4)
- **RVC Engine**: HuBERT 768 + RMVPE + FAISS IVF256 + NSF-HifiGAN

## Getting Started

See individual module READMEs:
- [`voice-model-server/README.md`](voice-model-server/README.md)
- [`voice-orchestrator/README.md`](voice-orchestrator/README.md)
- [`experiments/README.md`](experiments/README.md)

## License

MIT
