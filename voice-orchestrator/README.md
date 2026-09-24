# voice-orchestrator

Java 21 Spring Boot 3 orchestration layer — pipeline, cache, API, gRPC client.

## Responsibilities

- Expose REST + WebSocket API to novel reader frontend
- Split paragraphs into sentences
- Manage priority job queue (P0 → current, P1 → next 2, P2 → rest)
- Prefetch upcoming sentences in background
- Cache synthesized audio (Caffeine LRU + disk overflow)
- Call Python model server via gRPC (non-blocking)
- Handle circuit breaking, retry with backoff, graceful degradation

## Structure

```
voice-orchestrator/
├── src/main/
│   ├── java/com/novelreader/voice/
│   │   ├── VoiceOrchestratorApplication.java
│   │   ├── api/
│   │   │   ├── VoiceController.java         # REST endpoints
│   │   │   └── VoiceWebSocketHandler.java   # WebSocket streaming
│   │   ├── pipeline/
│   │   │   ├── PipelineOrchestrator.java    # main Flux chain
│   │   │   ├── SentenceSplitter.java        # paragraph → sentences
│   │   │   ├── PriorityJobQueue.java        # PriorityBlockingQueue
│   │   │   └── PrefetchWorker.java          # background lookahead
│   │   ├── grpc/
│   │   │   ├── VoiceGrpcClient.java         # stub + retry + circuit breaker
│   │   │   └── GrpcExceptionMapper.java     # StatusException → HTTP
│   │   ├── cache/
│   │   │   ├── AudioCacheService.java       # Caffeine + disk overflow
│   │   │   └── CacheKeyBuilder.java         # hash(text+emotion+char)
│   │   ├── exception/
│   │   │   ├── GlobalExceptionHandler.java  # @ControllerAdvice
│   │   │   ├── StageResult.java             # sealed Success/Degraded/Failed
│   │   │   ├── ErrorCode.java               # typed error codes enum
│   │   │   └── FallbackChain.java           # per-error recovery logic
│   │   └── config/
│   │       ├── GrpcClientConfig.java        # channel pool, TLS, timeout
│   │       ├── CacheConfig.java             # Caffeine spec
│   │       ├── PipelineConfig.java          # thread pool sizes
│   │       └── ResilienceConfig.java        # circuit breaker + retry policy
│   └── proto/
│       └── voice_service.proto              # symlink to ../proto/
├── build.gradle
└── README.md
```

## Setup

```bash
./gradlew generateProto   # auto-generates Java stubs from .proto
./gradlew bootRun         # starts on port 8080
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server + GPU + circuit breaker status |
| `POST` | `/synthesize` | Single sentence → WAV |
| `POST` | `/synthesize/batch` | Full paragraph → all sentences |
| `WS` | `/stream` | WebSocket real-time audio streaming |

## Key Dependencies

- Spring Boot 3 + WebFlux (reactive)
- Project Reactor (Flux/Mono pipeline)
- gRPC Netty Shaded (non-blocking transport)
- Resilience4j (circuit breaker + retry)
- Caffeine (LRU cache)
- Java 21 Virtual Threads
