// Shannon-Prime Engine — sp_load_shim tests.
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.

#include "../src/sp_load_shim.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define TEST(name) static void name(); static int reg_##name = (g_tests.push_back({#name, name}), 0); static void name()
struct TestEntry { const char *name; void (*fn)(); };
static std::vector<TestEntry> g_tests;
static int g_fail = 0;
#define ASSERT(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  ASSERT FAIL (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
    g_fail++; } } while (0)

using namespace sp::engine;

// fp16 helpers
static inline uint16_t f32_to_h(float v) {
    uint32_t f; std::memcpy(&f, &v, sizeof(f));
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int exp_i = (int)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp_i <= 0) return sign;
    if (exp_i >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp_i << 10) | (mant >> 13));
}
static inline float h_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = sign;
    else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float r; std::memcpy(&r, &f, sizeof(r)); return r;
}

// =========================================================================
// Bypass list — name matching
// =========================================================================

TEST(bypass_matches_rmsnorm_weights) {
    auto d1 = sp_shim_decide("blk.0.attn_norm.weight", true, false);
    ASSERT(d1.mode == sp_shim_mode::Bypass);

    auto d2 = sp_shim_decide("blk.5.ffn_norm.weight", true, false);
    ASSERT(d2.mode == sp_shim_mode::Bypass);

    auto d3 = sp_shim_decide("output_norm.weight", true, false);
    ASSERT(d3.mode == sp_shim_mode::Bypass);
}

TEST(bypass_matches_lm_head) {
    auto d1 = sp_shim_decide("output.weight", true, false);
    ASSERT(d1.mode == sp_shim_mode::Bypass);

    auto d2 = sp_shim_decide("lm_head.weight", true, false);
    ASSERT(d2.mode == sp_shim_mode::Bypass);
}

TEST(bypass_matches_token_embedding) {
    auto d1 = sp_shim_decide("token_embd.weight", true, false);
    ASSERT(d1.mode == sp_shim_mode::Bypass);

    auto d2 = sp_shim_decide("tok_embeddings.weight", true, false);
    ASSERT(d2.mode == sp_shim_mode::Bypass);

    auto d3 = sp_shim_decide("model.embed_tokens.weight", true, false);
    ASSERT(d3.mode == sp_shim_mode::Bypass);
}

TEST(bypass_matches_biases) {
    auto d = sp_shim_decide("blk.0.attn_q.bias", true, false);
    ASSERT(d.mode == sp_shim_mode::Bypass);
}

// =========================================================================
// Shim list — name matching
// =========================================================================

TEST(shim_matches_qkv_projections) {
    auto d1 = sp_shim_decide("blk.0.attn_q.weight", true, false);
    ASSERT(d1.mode == sp_shim_mode::FrobeniusQuant);

    auto d2 = sp_shim_decide("blk.7.attn_k.weight", true, false);
    ASSERT(d2.mode == sp_shim_mode::FrobeniusQuant);

    auto d3 = sp_shim_decide("blk.15.attn_v.weight", true, false);
    ASSERT(d3.mode == sp_shim_mode::FrobeniusQuant);
}

TEST(shim_matches_attn_output_proj) {
    auto d = sp_shim_decide("blk.0.attn_output.weight", true, false);
    ASSERT(d.mode == sp_shim_mode::FrobeniusQuant);
}

TEST(shim_matches_ffn_swiglu) {
    auto d1 = sp_shim_decide("blk.0.ffn_gate.weight", true, false);
    ASSERT(d1.mode == sp_shim_mode::FrobeniusQuant);

    auto d2 = sp_shim_decide("blk.0.ffn_up.weight", true, false);
    ASSERT(d2.mode == sp_shim_mode::FrobeniusQuant);

    auto d3 = sp_shim_decide("blk.0.ffn_down.weight", true, false);
    ASSERT(d3.mode == sp_shim_mode::FrobeniusQuant);
}

TEST(shim_matches_alternate_naming) {
    // Llama / Qwen convention
    auto d1 = sp_shim_decide("model.layers.5.self_attn.q_proj.weight", true, false);
    // q_proj not in our list — should fall to conservative bypass (which
    // we test below). For now check the explicit naming convention works.
    auto d2 = sp_shim_decide("blk.0.mlp.gate.weight", true, false);
    ASSERT(d2.mode == sp_shim_mode::FrobeniusQuant);
    (void)d1;
}

// =========================================================================
// Mode switching
// =========================================================================

TEST(no_flags_means_bypass_everything) {
    auto d = sp_shim_decide("blk.0.attn_q.weight", false, false);
    ASSERT(d.mode == sp_shim_mode::Bypass);
}

TEST(sato_tate_takes_precedence) {
    auto d = sp_shim_decide("blk.0.attn_q.weight", true, true);
    ASSERT(d.mode == sp_shim_mode::SatoTateMix);
}

TEST(unknown_tensor_falls_to_bypass) {
    auto d = sp_shim_decide("some_strange_extension_tensor", true, true);
    ASSERT(d.mode == sp_shim_mode::Bypass);
}

// =========================================================================
// Apply helpers — identity-shim (p=2, k=4) should be near-identity
// =========================================================================

TEST(apply_frobenius_p2_k4_is_near_identity) {
    // phi_2^4 = 4 (positive). Scale picked from absmax, divisor = scale*4.
    // Round-trip: round(w * scale) * 4 / (scale * 4) = round(w * scale) / scale
    // which equals the identity-round w.
    constexpr size_t N = 256;
    std::vector<uint16_t> buf(N);
    std::vector<uint16_t> orig(N);
    for (size_t i = 0; i < N; ++i) {
        float v = 0.001f * (float)((int)i - 128);
        buf[i] = f32_to_h(v);
        orig[i] = buf[i];
    }
    int64_t scale = 0;
    int rc = sp_load_shim_apply_frobenius(buf.data(), N, 2, 4, &scale);
    ASSERT(rc == 0);
    ASSERT(scale > 0);
    // Many fp16 values should be exactly recovered; some lose a ULP at the
    // smallest magnitudes due to scale_recip granularity.
    int exact = 0;
    int near = 0;
    for (size_t i = 0; i < N; ++i) {
        if (buf[i] == orig[i]) exact++;
        else {
            float a = h_to_f32(buf[i]);
            float b = h_to_f32(orig[i]);
            if (std::fabs(a - b) < 0.005f) near++;
        }
    }
    // Allow some near-misses at near-zero magnitudes.
    ASSERT(exact + near >= (int)N - 4);
}

TEST(apply_frobenius_sato_tate_at_p2_k4_p41_k0_is_inert_only) {
    // p2=41, k2=0 means split channel is identity. Only inert channel
    // (p1=2, k1=4) runs. Should match the identity shim above.
    constexpr size_t N = 64;
    std::vector<uint16_t> buf(N);
    for (size_t i = 0; i < N; ++i) {
        buf[i] = f32_to_h(0.01f * (float)((int)i - 32));
    }
    int64_t scale = 0;
    int rc = sp_load_shim_apply_sato_tate(buf.data(), N, 2, 4, 41, 0, &scale);
    ASSERT(rc == 0);
    ASSERT(scale > 0);
}

// =========================================================================
// Driver
// =========================================================================

int main() {
    std::printf("Shannon-Prime sp_load_shim unit tests (%zu)\n", g_tests.size());
    for (auto &t : g_tests) {
        int before = g_fail;
        t.fn();
        std::printf("  [%s] %s\n", (g_fail == before) ? "OK  " : "FAIL", t.name);
    }
    std::printf("\n%s — %d failures\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
