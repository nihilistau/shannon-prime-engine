// test_sp_quant_q5k — bit-for-bit parity against ggml's
// dequantize_row_q5_K. Self-contained: synthesizes random fp32 input,
// quantizes via ggml's Q5_K encoder, then dequants the resulting bytes
// with BOTH our sp_dequant_q5_K_to_f32 and ggml's dequantize_row_q5_K
// and asserts the outputs are byte-equal.
//
// Tests our DECODE path matches ggml's. We deliberately don't test the
// encode/quantize path because forward_native consumes already-quantized
// weights from GGUF — encode is ggml-internal forever.
//
// Exit 0 on full bit-exact match, 1 on any mismatch.

#include "sp_quant.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main() {
    constexpr int N_BLOCKS = 16;                         // 16 × 256 = 4096 elements
    constexpr int N_ELEMS  = N_BLOCKS * sp::engine::SP_QK_K;

    // Synthesize random fp32 input with realistic transformer-weight
    // magnitude (roughly N(0, 0.02) — that's typical init / fine-tuned
    // distribution, well within Q5_K's representable range).
    std::vector<float> src(N_ELEMS);
    std::mt19937 rng(0xDEADBEEFu);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    for (auto& v : src) v = nd(rng);

    // Quantize via ggml. Block size from public type traits.
    const struct ggml_type_traits* tt = ggml_get_type_traits(GGML_TYPE_Q5_K);
    const size_t block_size = tt->blck_size;     // 256 elems/block
    const size_t type_size  = tt->type_size;     // 176 bytes/block

    std::vector<uint8_t> packed(N_BLOCKS * type_size);
    ggml_quantize_chunk(GGML_TYPE_Q5_K, src.data(), packed.data(),
                        /*start=*/0, /*nrows=*/1, /*n_per_row=*/N_ELEMS,
                        /*imatrix=*/nullptr);

    // Sanity: block layout matches our struct.
    if (block_size != (size_t)sp::engine::SP_QK_K
        || type_size != sizeof(sp::engine::sp_block_q5_K)) {
        std::fprintf(stderr,
            "ggml Q5_K block size mismatch: ggml=%zux%zuB, ours=%dx%zuB\n",
            block_size, type_size,
            sp::engine::SP_QK_K, sizeof(sp::engine::sp_block_q5_K));
        return 2;
    }

    // Dequant via both paths.
    std::vector<float> ours(N_ELEMS);
    std::vector<float> theirs(N_ELEMS);
    sp::engine::sp_dequant_q5_K_to_f32(
        (const sp::engine::sp_block_q5_K*)packed.data(),
        ours.data(),
        N_BLOCKS);
    // ggml's public API: type_traits.to_float reads `n` elements.
    tt->to_float(packed.data(), theirs.data(), (int64_t)N_ELEMS);

    // Bit-exact compare.
    size_t mismatches = 0;
    double max_diff = 0.0;
    for (int i = 0; i < N_ELEMS; ++i) {
        const float a = ours[i];
        const float b = theirs[i];
        if (a != b) {
            ++mismatches;
            const double d = std::fabs((double)a - (double)b);
            if (d > max_diff) max_diff = d;
            if (mismatches <= 5) {
                std::fprintf(stderr,
                    "  mismatch at i=%d: ours=%.9g  ggml=%.9g  diff=%.9g\n",
                    i, a, b, d);
            }
        }
    }

    std::fprintf(stderr,
        "test_sp_quant_q5k:\n"
        "  blocks      = %d (%d elements)\n"
        "  mismatches  = %zu / %d\n"
        "  max abs err = %.9g\n"
        "  result      = %s\n",
        N_BLOCKS, N_ELEMS, mismatches, N_ELEMS, max_diff,
        mismatches == 0 ? "PASS (bit-exact)" : "FAIL");

    return mismatches == 0 ? 0 : 1;
}
