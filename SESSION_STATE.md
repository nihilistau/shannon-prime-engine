# SESSION_STATE.md — Shannon-Prime Engine Live State
# Updated: 2026-05-07 | Phase: 9 Beast Canyon Desktop Engine SHIPPED

## Current Proven Metrics (Device-Validated)

| Metric | Value | How Proven |
|--------|-------|-----------|
| Prefill | 423–500+ tok/s | 302ms / 128-token chunk on S22U |
| Decode | **BLOCKED** | 5th graphExecute hangs (see Blockers) |
| Decode (historical) | ~20 tok/s (~49ms/step) | Was working in a prior session, code lost to uncommitted edits |
| Drain reset cost | ~200-500ms one-time | Backend+device teardown+recreate (prefill->decode only) |
| HTP Residency | 4 splits, graph-switching | ~1390MB limit (split 2+3 OOM if concurrent) |
| Coherence | nan=0 all splits | UFIXED_16 bridge + zero-point 30833 |
| Teardown | Clean (exit 0) | dlclose deferred |
| HTTP Server | Running on 0.0.0.0:8080 | OpenAI-compat + frontend served |
| Frontend | Functional in browser | Glassmorphic dark-mode, live status |

## DECODE STATUS: Regression — Needs Reconstruction

**Decode WAS working.** 128-token generation validated on S22U with correct output,
~20 tok/s, exit 0. The fix was `sp_qnn_drain_htp()` doing a backend+device reset
(backendFree + backendCreate) between prefill and decode.

**Current state:** The working code was overwritten by subsequent debugging attempts
without being committed first. The current drain_htp code uses the same documented
approach (backend+device reset) but something in the surrounding code differs from
the working state. The decode hang has returned.

**To reconstruct:** The drain_htp approach is correct. Focus on what ELSE changed
in sp_qnn.c and qnn_bin_driver.cpp since the working state. Diagnostic fprintf
calls were added to sp_qnn.c (context counters, pre-execute logging). The generate()
function in qnn_bin_driver.cpp was modified for speculative decode. Any of these
changes could affect the execution flow enough to trigger the hang.

**Diagnostic findings (if useful for reconstruction):**
- The hang manifests on the 5th graphExecute within a process
- contextCreate always succeeds; graphExecute hangs
- The 4-cycle pattern holds for same-split and cross-split tests
- The working version somehow avoided or worked around this pattern

## Hardware Constants (Rosetta Stone)

```
UFIXED_POINT_16 dtype    = 1046  (NOT fp16 = 534)
Embedding zero-point     = 30833 (fp32 ~ 7.2e-6 * uint16 - 0.222)
Cos/Sin scale            = 3.051804378628731e-05, offset = -32768
Mask scale               = 0.0015259021893143654, offset = -65535
Buffer alignment         = 4096 bytes (ION/SMMU page boundary)
HTP priority             = QNN_PRIORITY_NORMAL (decode contexts)
QNN SDK                  = 2.45.40.260406 (QAIRT)
Model                    = Qwen3-4B w4a16 (AI Hub-compiled, 4 splits)
Device                   = S22 Ultra (SM8450 / Waipio / Hexagon V69)
5th-execute limit        = HARD — after 4 create+execute+free cycles, 5th hangs
                           Confirmed same-split (not about switching graphs)
                           Confirmed same seq_len (not about decode vs prefill)
                           Survives dlclose+dlopen, backend+device reset
```

## What's Working Right Now

- [x] sp_qnn.c: dlopen, load_binary_list (shareResources), execute, destroy
- [x] qnn_bin_driver: 4-split pipeline with residual carry, UFIXED_16 encode/decode
- [x] qnn_bin_quant_table.h: all 12 layers + cos/sin/mask/embedding scales recovered
- [x] Persistent ION tensor registration (rpcmem-backed)
- [x] HTTP server with OpenAI-compat API (/v1/chat/completions, /v1/models)
- [x] Frontend served from --www directory, live status, hardware telemetry
- [x] Android build (NDK, aarch64, QNN+Hexagon enabled)
- [x] Windows CUDA build (desktop dev/bench)
- [x] CPU forward pass (ggml-backed, all compression modes)
- [x] KV cache: ship/sqfree/hier/dual modes
- [x] System 1/2 entropy-gated dual-cache
- [x] Runtime graph build (MatMul, KQ+Softmax) -- Phase 2.5
- [x] Phase 8: SpOracle + speculative decode loop -- builds clean Android + Windows
- [x] Phase 8+: MoE curriculum + Top-2 prefetch + confidence gate -- SHIPPED v0.7.0
- [x] Phase 9: Beast Canyon desktop engine -- SHIPPED v0.8.0
  - [x] Optane reservoir (mmap GGUF, expert pointer table, stride audit)
  - [x] AVX-512 Shredder (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K/F16 → fp16 staging)
  - [x] Heterogeneous orchestrator (dual-GPU, ping-pong buffers, sidecar)
  - [x] Cross-GPU sync barriers (CUDA event + Vulkan fence + pre-shred callback)
  - [x] Diagnostics pulse monitor (stride/PCIe/barrier latency tracking)
  - [x] `--beast <gguf>` CLI flag + `beast_test` verb
  - [x] Verified: Dolphin-1B Q4_K_M — 148 tensors, 2.88 ms boot, 1.56 µs stride
  - [x] Dual-GPU auto-detection: RTX 2060 (12 GB CUDA) + Intel UHD (16 GB Vulkan)
  - [x] CUDA stream/event + Vulkan instance auto-provisioned at boot
  - [x] MinGW cudart linkage: gendef+dlltool import lib from DLL (no nvcc needed)
- [x] CLI unified flag parser -- all verbs accept all engine Config flags (a379442)
- [x] Prefill (single forward pass, 4 splits) -- works every time
- [ ] **Decode (multi-token generation) -- BLOCKED by 5th-execute hang**

## What's Next (Priority Order)

### URGENT: Fix Decode Hang (5th graphExecute)
The 4-cycle limit must be solved before any multi-token generation works.
Options to explore:
1. **Reconstruct the working drain_htp** -- the fix existed, was validated at 128
   tokens, but the code was lost. Git blame won't help (never committed).
2. **Fork-based decode** -- child process for each decode step (new PID = fresh /dev/cdsp)
3. **Reduce prefill to <4 cycles** -- e.g., load_binary_list for parallel context load
4. **2-split model** -- re-export Qwen3-4B as 2 splits instead of 4; prefill = 2 cycles,
   decode starts at cycle 3 (under the limit)
5. **Contact Qualcomm** -- this may be a known QNN HTP backend bug with a workaround

### Phase 6: HVX Logit Path (COMPLETE -- 2026-05-04)
- HVX argmax kernel live on cDSP: `libsp_hex_skel.so` loads on domain 3 via FastRPC
- rpcmem logit buffer: `out_bufs[logits_idx]` allocated via `rpcmem_alloc` (zero-copy ION)

### Phase 7: DMA Probe + Testsig (COMPLETE -- 2026-05-05) [BLOCKED on unsigned PD]
- All Hexagon DMA paths require signed PD on S22U
- Weight streaming deferred until signed PD / testsig available

### Phase 8: NEON Oracle (CODE COMPLETE -- 2026-05-05) [BLOCKED on decode hang]
- `src/speculative_oracle.h/cpp` -- SpOracle class wrapping ForwardNativeContext
- `QnnBinSession::generate()` -- speculative decode loop with batch_verify_step
- CLI: `qnn_oracle_bench` and `qnn_bin_generate` verbs
- Cannot validate on device until decode hang is fixed

### Phase 8+: MoE Expert Curriculum + Top-2 Speculative Prefetch (SHIPPED -- 2026-05-07)
- `sp_moe_curriculum.h` -- EWMA heatmap, curriculum pulse, tier assignment
- `sp_prefetch_engine.h` -- dual-slot Top-2 speculative prefetch + confidence gate
- Validated on Qwen3.6-35B-A3B (256 experts, top-8, 40 layers) -- desktop CPU
- Validated graceful decline on dense models (Qwen3-0.6B)
- CLI: `--moe-curriculum` flag enables full system
- Tag: `v0.7.0-moe-curriculum` (engine d1421f7, submodule a9b8d10)

### Phase 9: MoE Scaling (Qwen 3.6 27B)
- JIT expert streaming, HTP persistent context per active expert
- Blocked on Phase 8 validation + decode fix
- MoE curriculum now provides the expert heatmap for JIT prefetch decisions

## Blockers / Known Issues

- **5th graphExecute hang -- CRITICAL, blocks all multi-token generation**
- Alloc2 warnings from V69 driver -- cosmetic only
- Vulkan VK_ERROR_DEVICE_LOST on RTX 2060 -- tracked separately
- 1/188 test flake (synthetic-K-pipeline, random data edge case)
- Phase 8 KV greedy commit: n_past inconsistency after partial acceptance (Phase 8B fix)
- DMA weight streaming blocked on unsigned PD (Phase 7)

## Active Files Being Modified

| File | Last Changed | What |
|------|-------------|------|
| `lib/.../sp_qnn_runner/sp_qnn.c` | 2026-05-05 | drain_htp: currently dlclose+dlopen (DOES NOT FIX HANG) |
| `lib/.../sp_qnn_runner/sp_qnn.h` | 2026-05-05 | Updated drain_htp doc comment |
| `src/speculative_oracle.h` | 2026-05-05 | Phase 8 SpOracle class header |
| `src/speculative_oracle.cpp` | 2026-05-05 | ForwardNativeContext wrapper + NEON argmax |
| `src/qnn_bin_driver.h` | 2026-05-05 | Added set_oracle() + batch_verify comments |
| `src/qnn_bin_driver.cpp` | 2026-05-05 | batch_verify_step + speculative decode loop + drain call |
| `src/cli/main.cpp` | 2026-05-05 | qnn_oracle_bench + qnn_bin_generate verbs |
| `CMakeLists.txt` | 2026-05-05 | speculative_oracle.cpp added to sp_engine |

## Diagnostic Commands

```bash
# Prefill-only test (always works, 4 cycles)
adb shell "LD_LIBRARY_PATH=/data/local/tmp/sp-engine:/data/local/tmp/sp22u/qnn \
  ADSP_LIBRARY_PATH='/data/local/tmp/sp-engine;/vendor/lib/rfsa/adsp' \
  /data/local/tmp/sp-engine/sp-engine qnn_bin_run \
  --tokenizer /data/local/tmp/sp22u/model.gguf \
  --prompt 'The capital of France is' \
  /data/local/tmp/sp22u/qnn/qwen3_4b_*.bin"

# Multi-token test (WILL HANG at 5th graphExecute)
adb shell "LD_LIBRARY_PATH=/data/local/tmp/sp-engine:/data/local/tmp/sp22u/qnn \
  ADSP_LIBRARY_PATH='/data/local/tmp/sp-engine;/vendor/lib/rfsa/adsp' \
  /data/local/tmp/sp-engine/sp-engine qnn_bin_generate \
  --tokenizer /data/local/tmp/sp22u/model.gguf \
  --prompt 'The capital of France is' --n-predict 2 --head-dim 128 \
  /data/local/tmp/sp22u/qnn/qwen3_4b_*.bin"

# FastRPC memory stats
adb shell "cat /sys/kernel/debug/adsprpc/stats"
```

## Trap Avoidance Checklist

Before making changes, verify you're NOT:
- [ ] Retreating to ggml/llama.cpp standard paths
- [ ] Treating UFIXED_16 as fp16
- [ ] Using zero-initialized KV buffers (need zero-point 30833)
- [ ] Using unaligned buffers (need 4096-byte alignment)
- [ ] Declaring the 1.5GB HTP limit real (it's debunked)
- [ ] Optimizing invariants without understanding them
- [ ] Mixing code from different build paths (frankenpatch)
- [ ] Suggesting wrapping up or sleeping
- [ ] **Rebooting the phone** (NEVER run adb reboot)
- [ ] **Overwriting working code without committing first**
