# KSTE Calibration Ledger

**Phase 4 of the Friedman Stack roadmap.  Records every PPL calibration sweep against Gemma3-1B on WikiText-103.**

---

## Methodology

The KSTE encoder ships with three runtime knobs that move the trade-off between sieve aggressiveness and PPL drift:

| Knob          | What it controls                                 | Range under sweep |
|---------------|--------------------------------------------------|-------------------|
| `tau_A`       | anchor inclusion threshold (× amax)              | 0.0 .. 0.05       |
| `alpha`       | residual-to-anchor attachment ratio              | 0.3 .. 0.9        |
| `capacity`    | per-(layer, head) Friedman cache slot count      | 1024 .. 4096      |

For each grid point we run `sp-engine.exe perplexity-sp` twice:

```bash
# Baseline
sp-engine.exe perplexity-sp \
    --model gemma3-1b.gguf --corpus wikitext-103-valid.txt \
    --ctx 2048 --chunks 4 --threads 16 \
    --frobenius-quant -p 41 -k 8 \
    --poly-attn --ntt-crt

# With sieve
... + --friedman-sieve --friedman-mode policy \
      --friedman-capacity <cap> \
      --kste-tau-A <tau> --kste-alpha <alpha>
```

and record the per-config:

```json
{
  "cap": <int>, "tau_A": <float>, "alpha": <float>,
  "ppl_sieve": <float>, "ppl_baseline": <float>,
  "delta_pct": <float>,           // (sieve - baseline) / baseline
  "eviction_rate": <float>,       // fraction of tokens evicted by sieve
  "wall_s": <float>,              // wall-time for the sieve run
  "passes_T2_3": <bool>,          // |delta_pct| <= 0.005
  "passes_T2_2": <bool>           // eviction_rate >= 0.20
}
```

`scripts/calibrate_kste.py` runs the sweep and appends a summary row below.

## Gates

- **T2.3 (the ship gate).**  `|delta_pct| <= 0.5%`.  Strict — see roadmap §4 risk.  If no grid point satisfies this, STOP and report; do not ship the sieve as default.
- **T2.2.**  `eviction_rate >= 20%` at steady state.  Necessary for the sieve to be doing useful work.  If T2.3 passes but T2.2 fails, the sieve is "harmless but trivial" (no value).

The best calibration is the **T2.3-passing point with the highest eviction rate.**  Ties broken on lowest `delta_pct`.

## Pre-flight gate: encoder resolution

Before running any PPL sweep, verify the encoder is producing meaningful subsumption:

```bash
tests/build/test_sp_kste_resolution  # writes T4_RES_PROBE.json
```

T4_RES_PROBE measures, on synthetic clustered K-vectors, whether intra-cluster trees actually embed into one another (and inter-cluster ones don't).  **If `intra_embed_rate` is < 0.05 at sigma=0.05, the encoder is over-discriminating and the PPL sweep will be meaningless** — eviction rate will be near zero and PPL delta will be ~0, but T2.2 will fail.  Iterate the encoder (Phase 4b remediation in SESSION-STATE-friedman-4.md) before sweeping.

## Sweeps

(rows appended by `scripts/calibrate_kste.py`)
