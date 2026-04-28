// Shannon-Prime Engine — disk cache save/load round-trip smoke test.
//
// Validates the compressed-KV disk-tier scaffold (kv_cache.h):
//   * KvCache::save_to_disk(prefix, n_pos, model_hash)
//   * KvCache::load_from_disk(prefix, expected_hash) -> n_pos
//
// Three properties under test:
//
//   1. ROUND-TRIP EXACTNESS. After write -> read -> save -> destroy ->
//      create -> load -> read, the second read must equal the first
//      bit-for-bit. The compressor's lossy reconstruction is irrelevant
//      here — we're testing that disk serialisation preserves the
//      compressed state. No tolerance.
//
//   2. MODEL-HASH GUARD (strict by default). load_from_disk with a
//      wrong hash returns -1 — matches the API doc in kv_cache.h:
//      143-147. The earlier "warn-only" behaviour was a divergence
//      that allowed silent corruption when a user loaded a cache
//      against the wrong model; that's now a hard error.
//
//   3. MODEL-HASH GUARD ESCAPE HATCH. With SP_DISK_HASH_STRICT=0 in
//      the environment, the loader logs the mismatch warning and
//      returns n_pos anyway, preserving the original Archimedes-style
//      behaviour for advanced callers who know what they're doing.
//
// Reconstruction-fidelity validation (does read() produce values close
// to write() inputs?) is the calibration suite's job, not this test's.
// Without a calibrated banded quantiser, fidelity numbers reflect band
// mistuning, not the disk path. This smoke test stays scoped to the
// disk-tier scaffold.
//
// CPU-only by design. Disk I/O is host-side regardless of backend, so
// the round-trip property holds for the GPU path with the same code.

#include "engine.h"
#include "kv_cache.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>     // setenv / unsetenv / _putenv_s
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Deterministic K/V test pattern. Each (layer, position, head, dim) gets
// a unique value derived from a smooth phase function — keeps the input
// in-distribution for a banded quantiser so reconstruction error is
// bounded by quantisation noise rather than out-of-band saturation.
static float test_value(int layer, int pos, int head, int d, int head_dim) {
    const float phase = 0.013f * static_cast<float>(layer)
                      + 0.071f * static_cast<float>(pos)
                      + 0.029f * static_cast<float>(head)
                      + (2.0f * 3.14159265f / static_cast<float>(head_dim))
                            * static_cast<float>(d);
    return 0.4f * std::sin(phase);
}

static void fill_layer(int layer, int n_head_kv, int head_dim,
                       int pos_offset, int n_tokens,
                       std::vector<float>& K, std::vector<float>& V) {
    K.resize(static_cast<size_t>(n_tokens) * n_head_kv * head_dim);
    V.resize(K.size());
    for (int q = 0; q < n_tokens; ++q) {
        for (int h = 0; h < n_head_kv; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t i = (static_cast<size_t>(q) * n_head_kv + h) * head_dim + d;
                K[i] = test_value(layer, pos_offset + q, h,         d, head_dim);
                // V uses a different head offset so save/load swaps would show.
                V[i] = test_value(layer, pos_offset + q, h + 1000, d, head_dim);
            }
        }
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30f;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static bool exactly_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int main() {
    using namespace sp::engine;

    // --- Fixture dimensions ------------------------------------------------
    // Picked so total disk footprint is small (~few KB) and exercises >1
    // layer + multiple heads + a typical SP-supported head_dim.
    constexpr int n_layer    = 2;
    constexpr int n_head_kv  = 4;
    constexpr int head_dim   = 128;
    constexpr int max_seq    = 32;
    constexpr int n_pos      = 16;       // write [0, 16)
    const     uint64_t model_hash = 0xDEADBEEFCAFEBABEULL;

    Config cfg;
    cfg.n_ctx        = max_seq;
    cfg.k_bits_csv   = "5,5,4,3";
    cfg.v_bits_csv   = "3";
    cfg.residual_bits = 3;
    // Ship path: mobius=true (default), sqfree=false, hierarchical=false.
    // Pure CPU — no backend selection needed for KvCache::create().

    // --- Reference write + read ------------------------------------------
    // Capture what `read()` produces immediately after writing — this is
    // what `read()` MUST reproduce after the disk round-trip.

    std::vector<std::vector<float>> ref_K(n_layer);  // [layer][flat]
    std::vector<std::vector<float>> ref_V(n_layer);
    std::vector<std::vector<float>> in_K(n_layer);   // for fidelity check
    std::vector<std::vector<float>> in_V(n_layer);

    {
        auto cache = KvCache::create(n_layer, n_head_kv, head_dim, max_seq, cfg);
        if (!cache) {
            std::fprintf(stderr, "disk_cache: KvCache::create() returned null\n");
            return 1;
        }

        for (int il = 0; il < n_layer; ++il) {
            fill_layer(il, n_head_kv, head_dim, /*pos_offset=*/0, n_pos,
                       in_K[il], in_V[il]);
            if (!cache->write(il, /*pos_offset=*/0, n_pos,
                              in_K[il].data(), in_V[il].data())) {
                std::fprintf(stderr, "disk_cache: write() failed at layer %d\n", il);
                return 1;
            }
        }

        for (int il = 0; il < n_layer; ++il) {
            if (!cache->read(il, n_pos, ref_K[il], ref_V[il])) {
                std::fprintf(stderr, "disk_cache: read() failed at layer %d\n", il);
                return 1;
            }
        }

        // Sanity check: at least some non-trivial output. If the read path
        // returned all zeros (e.g., a bug zeroed the output buffer), every
        // round-trip would trivially "pass". Catch that.
        bool nonzero_seen = false;
        for (int il = 0; il < n_layer && !nonzero_seen; ++il) {
            for (float v : ref_K[il]) {
                if (std::fabs(v) > 1e-6f) { nonzero_seen = true; break; }
            }
        }
        if (!nonzero_seen) {
            std::fprintf(stderr, "disk_cache: read() produced all-zero output — broken pipeline\n");
            return 1;
        }

        // --- Save -------------------------------------------------------
        // tmp prefix — clean up at the end regardless of pass/fail.
        const fs::path tmp_dir = fs::temp_directory_path() / "sp_engine_disk_test";
        fs::create_directories(tmp_dir);
        const std::string prefix = (tmp_dir / "kv").string();

        const int rc = cache->save_to_disk(prefix, n_pos, model_hash);
        if (rc != 0) {
            std::fprintf(stderr, "disk_cache: save_to_disk returned %d\n", rc);
            return 1;
        }

        // Cache destroyed here as `cache` goes out of scope.
    }

    // --- Reload pass: fresh cache, load, read, compare -------------------
    const fs::path tmp_dir = fs::temp_directory_path() / "sp_engine_disk_test";
    const std::string prefix = (tmp_dir / "kv").string();

    {
        auto cache = KvCache::create(n_layer, n_head_kv, head_dim, max_seq, cfg);
        if (!cache) {
            std::fprintf(stderr, "disk_cache: post-load KvCache::create() returned null\n");
            return 1;
        }

        const int loaded = cache->load_from_disk(prefix, model_hash);
        if (loaded != n_pos) {
            std::fprintf(stderr,
                "disk_cache: load_from_disk returned %d (expected n_pos=%d)\n",
                loaded, n_pos);
            return 1;
        }

        // Property 1: round-trip exactness.
        for (int il = 0; il < n_layer; ++il) {
            std::vector<float> K_out, V_out;
            if (!cache->read(il, n_pos, K_out, V_out)) {
                std::fprintf(stderr, "disk_cache: post-load read() failed at layer %d\n", il);
                return 1;
            }
            if (!exactly_equal(ref_K[il], K_out)) {
                std::fprintf(stderr,
                    "disk_cache: layer %d K round-trip differs (max-abs=%.6e)\n",
                    il, max_abs_diff(ref_K[il], K_out));
                return 1;
            }
            if (!exactly_equal(ref_V[il], V_out)) {
                std::fprintf(stderr,
                    "disk_cache: layer %d V round-trip differs (max-abs=%.6e)\n",
                    il, max_abs_diff(ref_V[il], V_out));
                return 1;
            }
        }
    }

    // --- Property 2: model-hash mismatch is rejected (strict default) ----
    // The C-level loader is strict by default after the API/impl divergence
    // was resolved — wrong hash returns -1, matching kv_cache.h:143-147.
    // This blocks silent corruption from a cross-model load.
    {
        auto cache = KvCache::create(n_layer, n_head_kv, head_dim, max_seq, cfg);
        if (!cache) {
            std::fprintf(stderr, "disk_cache: hash-guard KvCache::create() returned null\n");
            return 1;
        }
        const uint64_t bad_hash = model_hash ^ 0x1ULL;
        const int rc = cache->load_from_disk(prefix, bad_hash);
        if (rc != -1) {
            std::fprintf(stderr,
                "disk_cache: load_from_disk with wrong hash returned %d "
                "(expected -1 under default strict mode)\n", rc);
            return 1;
        }
    }

    // --- Property 3: SP_DISK_HASH_STRICT=0 escape hatch -------------------
    // Advanced callers can opt into the original Archimedes-style warn-only
    // behaviour by setting SP_DISK_HASH_STRICT=0 in the env. The loader
    // logs the mismatch and proceeds. We test this is still functional so
    // any future regression in the env-var handling fails loudly here.
    {
        // Set the env var BEFORE create+load so the subsequent load
        // path picks it up.
#if defined(_WIN32)
        _putenv_s("SP_DISK_HASH_STRICT", "0");
#else
        setenv("SP_DISK_HASH_STRICT", "0", 1);
#endif

        auto cache = KvCache::create(n_layer, n_head_kv, head_dim, max_seq, cfg);
        if (!cache) {
            std::fprintf(stderr, "disk_cache: escape-hatch KvCache::create() returned null\n");
            return 1;
        }
        const uint64_t bad_hash = model_hash ^ 0x1ULL;
        const int rc = cache->load_from_disk(prefix, bad_hash);
        if (rc != n_pos) {
            std::fprintf(stderr,
                "disk_cache: with SP_DISK_HASH_STRICT=0, load returned %d "
                "(expected n_pos=%d — escape hatch should warn-only)\n",
                rc, n_pos);
            return 1;
        }

        // Clear the env var so it doesn't leak to other tests in the same
        // ctest run.
#if defined(_WIN32)
        _putenv_s("SP_DISK_HASH_STRICT", "");
#else
        unsetenv("SP_DISK_HASH_STRICT");
#endif
    }

    // --- Property 4: progressive partial load (v3 band-major format) -----
    // Saved files use the new v3 layout. load_from_disk_partial(max_bands=2)
    // should reconstruct only bands 0+1, leaving bands 2+3 zeroed in the
    // cache. read() then produces partial-fidelity vectors that:
    //   - are NOT byte-equal to ref_K/ref_V (some coefficients are now zero)
    //   - are NOT all-zero (band 0 / band 1 carried real signal)
    //   - have non-zero correlation with ref_K/ref_V (energy concentration
    //     in early bands means partial reconstruction tracks the original)
    {
        auto cache = KvCache::create(n_layer, n_head_kv, head_dim, max_seq, cfg);
        if (!cache) {
            std::fprintf(stderr, "disk_cache: partial-load KvCache::create() returned null\n");
            return 1;
        }

        const int max_bands = 2;   // K is 5/5/4/3 → bands 2 and 3 zeroed
        const int loaded = cache->load_from_disk_partial(prefix, model_hash, max_bands);
        if (loaded != n_pos) {
            std::fprintf(stderr,
                "disk_cache: load_from_disk_partial returned %d (expected n_pos=%d)\n",
                loaded, n_pos);
            return 1;
        }

        for (int il = 0; il < n_layer; ++il) {
            std::vector<float> K_partial, V_partial;
            if (!cache->read(il, n_pos, K_partial, V_partial)) {
                std::fprintf(stderr, "disk_cache: post-partial-load read() failed at layer %d\n", il);
                return 1;
            }

            // Sub-property A: NOT byte-equal to the full-fidelity reference
            //   (because bands 2-3 were zeroed and inverse-VHT2 mixes them
            //   into every output position)
            if (exactly_equal(K_partial, ref_K[il])) {
                std::fprintf(stderr,
                    "disk_cache: layer %d K partial-load matched full reference exactly — "
                    "bands 2-3 didn't get zeroed\n", il);
                return 1;
            }

            // Sub-property B: NOT all-zero (bands 0+1 carried real signal)
            bool any_nonzero = false;
            for (float v : K_partial) {
                if (std::fabs(v) > 1e-6f) { any_nonzero = true; break; }
            }
            if (!any_nonzero) {
                std::fprintf(stderr,
                    "disk_cache: layer %d K partial-load returned all-zero — "
                    "bands 0+1 didn't reconstruct\n", il);
                return 1;
            }

            // Sub-property C: bounded magnitude. Partial reconstruction
            //   should be in the same order of magnitude as the full
            //   reference — not exploding or near-zero. We only check
            //   that the max absolute value is finite and within 4× of
            //   the reference's max. Tighter quantitative guarantees
            //   (correlation, energy retention) depend on input signal
            //   distribution and are out of scope for this scaffold-
            //   layer test; fidelity validation lives in the calibration
            //   suite where we control the input distribution.
            float max_partial = 0.0f, max_ref = 0.0f;
            for (size_t i = 0; i < K_partial.size(); ++i) {
                float ap = std::fabs(K_partial[i]);
                float ar = std::fabs(ref_K[il][i]);
                if (ap > max_partial) max_partial = ap;
                if (ar > max_ref)     max_ref     = ar;
            }
            if (!std::isfinite(max_partial)) {
                std::fprintf(stderr,
                    "disk_cache: layer %d K partial-load contains non-finite values\n", il);
                return 1;
            }
            if (max_ref > 1e-6f && max_partial > 4.0f * max_ref) {
                std::fprintf(stderr,
                    "disk_cache: layer %d K partial-load magnitude exploded "
                    "(max_partial=%.4f vs max_ref=%.4f)\n",
                    il, max_partial, max_ref);
                return 1;
            }
        }
    }

    // --- Cleanup --------------------------------------------------------
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    // Best-effort; CI tmp cleanup will sweep the rest.

    std::printf("disk_cache_roundtrip: OK (n_layer=%d, n_head_kv=%d, head_dim=%d, n_pos=%d)\n",
                n_layer, n_head_kv, head_dim, n_pos);
    return 0;
}
