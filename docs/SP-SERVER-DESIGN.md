# SP-Server Design — OpenAI-Compatible Inference Server for Shannon-Prime

**Status:** Phase 0 (design lock). Implementation kicks off in Phase 2.
**Companion to:** `THEORY-FIRST-ENGINE-DESIGN.md`

## Goal

`sp-server` is a drop-in replacement for `llama-server`, `vllm.entrypoints.openai.api_server`, and `ollama serve`. It exposes the standard OpenAI v1 API and adds SP-specific endpoints for Frobenius quantization configuration and KV-cache introspection.

## API surface (Phase 2 target)

### Standard OpenAI v1

| Endpoint | Method | Behavior |
|--|--|--|
| `/v1/models` | GET | List loaded models (one or many) |
| `/v1/models/{id}` | GET | Single-model metadata |
| `/v1/completions` | POST | Text completion (legacy) |
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming) |
| `/v1/embeddings` | POST | Embedding extraction |
| `/health` | GET | Liveness probe |
| `/v1/health/ready` | GET | Readiness probe |

### SP-specific (additive)

| Endpoint | Method | Behavior |
|--|--|--|
| `/sp/v1/quant/config` | GET | Active quantization tier (Config A/B/C/D/E) |
| `/sp/v1/quant/set` | POST | Switch tier at runtime (model reload if needed) |
| `/sp/v1/kv/stats` | GET | KV cache: size, compression ratio, Hasse-Weil bound saturation |
| `/sp/v1/poncelet/depth` | GET | Per-request Poncelet adaptive depth count (last N requests) |
| `/sp/v1/oracle/predict` | POST | Activation oracle prefetch (debugging / instrumentation) |

## Request / response shapes (mirror OpenAI)

```jsonc
// POST /v1/chat/completions  (non-streaming)
{
    "model": "phi-3-mini-4k-sp-fp10",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "stream": false,
    "sp_quant": {                 // SP-specific extensions, optional
        "tier": "sato_tate_fp10", // "fp16", "fp8_frobenius", "sato_tate_fp10", "fp4"
        "p1": 2, "k1": 2,
        "p2": 41, "k2": 8
    }
}

// Response
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "created": 1747234567,
    "model": "phi-3-mini-4k-sp-fp10",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! ..."},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 24,
        "total_tokens": 32
    },
    "sp_diagnostics": {
        "kv_compression_ratio": 6.2,
        "poncelet_depth": 24,
        "active_quant_tier": "sato_tate_fp10"
    }
}
```

Streaming uses SSE with `data: <json>\n\n` chunks per the OpenAI spec.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ sp_server                                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ HTTP/1.1 + WS  (cpp-httplib, existing dependency)      │ │
│ └────────┬────────────────────────────────────────────────┘ │
│          ▼                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Request router                                          │ │
│ │   /v1/chat/completions  → ChatCompletionHandler         │ │
│ │   /v1/completions       → CompletionHandler             │ │
│ │   /v1/models            → ModelsHandler                 │ │
│ │   /sp/v1/*              → SPSpecificHandler             │ │
│ └────────┬────────────────────────────────────────────────┘ │
│          ▼                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Continuous batching scheduler (sp_batch.cpp)            │ │
│ │ - Active request slots                                  │ │
│ │ - Per-token round-robin                                 │ │
│ │ - Paged KV cache (SP-Frobenius compressed)              │ │
│ └────────┬────────────────────────────────────────────────┘ │
│          ▼                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ sp_forward (per-token, batched)                         │ │
│ │ - O_K-coordinate tensors                                │ │
│ │ - Frobenius-quantized weights                           │ │
│ │ - Sato-Tate mixed-precision (Config E)                  │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key design choices

### 1. Single binary, multiple subcommands

```
sp-engine generate --model model.gguf --prompt "Hello"
sp-engine bench    --model model.gguf --runs 10
sp-engine server   --model model.gguf --host 0.0.0.0 --port 8080
sp-engine quantize --in model.gguf --out model.sp.gguf --tier sato_tate_fp10
```

Same binary as `sp-engine` from Phase 1, just dispatches based on the first arg.

### 2. Streaming via SSE

```cpp
// sp_server.cpp pseudocode
void ChatCompletionHandler::handle_streaming(const Request& req, Response& res) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream",
        [&](size_t /*offset*/, DataSink& sink) {
            sp_request sp_req = parse_chat_request(req.body);
            sp_batch_slot slot = scheduler->add(sp_req);
            while (!slot.done) {
                const sp_token tok = slot.next_token();
                std::string chunk = format_sse_chunk(tok);
                sink.write(chunk.c_str(), chunk.size());
                if (!slot.streaming_ok()) break;
            }
            sink.write("data: [DONE]\n\n", 14);
            sink.done();
            return true;
        });
}
```

### 3. Continuous batching with Frobenius-compressed KV

Each request occupies a "slot" with its own KV history. The scheduler processes one token per slot per forward pass. Because SP-Frobenius compresses KV by 6.2× (Paper B §6.1), we can fit ~6× more slots than vLLM's standard fp16 paged KV cache at the same memory budget.

### 4. Chat templates as data

```cpp
// src/sp_chat_template.h
struct sp_chat_template {
    std::string name;             // "chatml", "llama3", "qwen3", "phi3", "gemma"
    std::string system_prefix;
    std::string system_suffix;
    std::string user_prefix;
    std::string user_suffix;
    std::string assistant_prefix;
    std::string assistant_suffix;
    std::string eos_token;
};

// Selectable via --chat-template flag, with auto-detection from GGUF metadata.
```

### 5. SP diagnostics in every response

The `sp_diagnostics` block on every chat/completion response gives operators visibility into:
- KV compression ratio achieved on this request
- Poncelet adaptive-depth count (how many layers actually ran)
- Active quantization tier
- Cache-hit rate (oracle prediction)

This is sp-server's superpower over llama-server: every request comes with proof that the framework is working as predicted by Papers A/B/C/D.

## Implementation status

Phase 0 (this commit): design doc + existing `http_server.cpp` skeleton (non-streaming, /v1/chat/completions + /v1/models only).

Phase 2 deliverable: full implementation per the API surface above. The existing `http_server.cpp` (~1000 LOC) becomes the basis; we extend it with SSE streaming, continuous batching, the `/sp/v1/*` namespace, and the model-management improvements.

## Comparison with reference implementations

| Aspect | llama-server | vLLM | ollama | **sp-server** |
|--|--|--|--|--|
| Lines of code | ~5K | ~100K | ~30K | ~5K (target) |
| Memory per slot at 8K ctx (fp16) | 1.4 GiB | 1.4 GiB paged | 1.4 GiB | **226 MiB** (6.2× SP compression) |
| First-token latency (cold) | ~200ms | ~150ms | ~250ms | target ≤200ms |
| Throughput at batch 32 | varies | best | mid | target match vLLM |
| OpenAI v1 compat | ✓ | ✓ | ✓ | ✓ (Phase 2) |
| Calibration-free fp8 | — | — | — | **✓** (Frobenius) |
| KV cache provability | — | — | — | **✓** (exact Möbius) |

## Risks and mitigations

| Risk | Mitigation |
|--|--|
| HTTP/2 + WebSocket complexity | Phase 2 ships HTTP/1.1 + SSE only (sufficient for OpenAI compat); HTTP/2 in Phase 5 |
| Tokenizer correctness across model families | Use sentencepiece for Llama/Phi-3, tiktoken-rs port for Qwen/GPT; per-model test vectors |
| Batching deadlock on long-prompt requests | Per-slot timeout + LRU eviction with KV-replay flag |
| Frobenius quant model not validated on Phi-3 | Paper D §4.3 pass/fail criteria; CI block on failure |

## Open questions for review

1. **Authentication.** Phase 2 ships no auth (assume reverse proxy). Phase 3 may add Bearer-token. OpenAI keys are out of scope.
2. **Multi-model serving.** Phase 2: single model per process. Phase 3: multi-model via subprocess pool. Phase 5: in-process multi-model with weight hot-swap.
3. **Function/tool calling.** Phase 2 ships pass-through; the model itself decides. Phase 3 adds parsing-aware sampler integration.

---

Phase 2 implementation begins after Phase 1 (theory-first forward pass) lands and validates Phi-3 perplexity within 1% of llama.cpp baseline.
