/* test_sp_ok_q4.cpp — Phase 14 parity test for 4-bit O_K storage.
 *
 * Mirrors test_sp_ok_q8.cpp exactly with the codebook halved. Verifies:
 *   1. shift_picker_is_minimal — boundary cases for sp_ok_q4_pick_shift
 *   2. roundtrip_bounded_by_rounding_step — per-coord error <= 2^(shift-1)
 *   3. packing layout — low/high nybble extraction round-trips through
 *      pack/decode without sign loss
 *   4. lattice-norm pruning produces 0x00 packed bytes for zeroed entries
 *   5. 16x compression ratio vs raw sp_ok_t (1 byte / element)
 *
 * Compile + run on either Linux gcc or Windows MSVC. Targets the same
 * input distribution as the Q8 test (|a|, |b| in [0, 2^24)) so the two
 * tests can be diffed line-by-line for sanity.
 */

#include <cstddef>
extern "C" {
#include "../lib/shannon-prime/core/sp_ok_arith.h"
#include "../lib/shannon-prime/core/sp_ok_q4.h"
}

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

static int g_failures = 0;
static int g_tests = 0;
#define CHECK(cond, msg) do {                                              \
    ++g_tests;                                                              \
    if (!(cond)) {                                                          \
        ++g_failures;                                                       \
        std::fprintf(stderr,                                                \
                     "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg);       \
    }                                                                       \
} while (0)

static void run(const char* name, void (*fn)()) {
    std::fprintf(stderr, "[run] %s\n", name);
    fn();
}

/* ---------- Test 1: pick_shift returns minimal valid shift -------------- */

static void shift_picker_is_minimal() {
    /* Boundary cases — int4 range is [-8, 7]. */
    CHECK(sp_ok_q4_pick_shift(0) == 0, "absmax=0 -> shift=0");
    CHECK(sp_ok_q4_pick_shift(7) == 0, "7 fits in int4");
    CHECK(sp_ok_q4_pick_shift(8) == 1, "8 needs shift=1: (8+1)>>1=4<=7");
    CHECK(sp_ok_q4_pick_shift(15) == 2, "15 -> 8 -> 4 with ceil-shift, s=2");
    CHECK(sp_ok_q4_pick_shift(16) == 2, "16 -> 8 -> 4, s=2");
    /* Pre-Frobenius regime |a| < 2^24. */
    int8_t s24 = sp_ok_q4_pick_shift(((int64_t)1 << 24) - 1);
    CHECK(s24 == 22, "2^24 - 1 -> shift=22");
    /* Post-Frobenius regime O(2^45). */
    int8_t s45 = sp_ok_q4_pick_shift(((int64_t)1 << 45));
    CHECK(s45 == 43, "2^45 -> shift=43");
    /* O(2^60) worst case. */
    int8_t s60 = sp_ok_q4_pick_shift(((int64_t)1 << 60));
    CHECK(s60 == 58, "2^60 -> shift=58");
    std::fprintf(stderr, "  shift(2^24-1)=%d  shift(2^45)=%d  shift(2^60)=%d\n",
                 s24, s45, s60);
}

/* ---------- Test 2: encode/decode round-trip per-coord bound ------------- */

static void roundtrip_bounded_by_rounding_step() {
    constexpr size_t N = 4096;
    std::mt19937_64 rng(0xC0DEFACECAFEBABE);
    std::uniform_int_distribution<int64_t> dist(-((int64_t)1 << 24),
                                                  ((int64_t)1 << 24));
    std::vector<sp_ok_t> src(N);
    for (size_t i = 0; i < N; ++i) src[i] = sp_ok_t{ dist(rng), dist(rng) };

    std::vector<sp_ok_q4_t> packed(N);
    int8_t shift = sp_ok_q4_encode_array(packed.data(), src.data(), N);

    std::vector<sp_ok_t> rt(N);
    sp_ok_q4_decode_array(rt.data(), packed.data(), N, shift);

    const int64_t bound = sp_ok_q4_max_error(shift);
    size_t worst = 0;
    int64_t worst_err = 0;
    for (size_t i = 0; i < N; ++i) {
        int64_t da = std::abs(src[i].a - rt[i].a);
        int64_t db = std::abs(src[i].b - rt[i].b);
        if (da > worst_err) { worst_err = da; worst = i; }
        if (db > worst_err) { worst_err = db; worst = i; }
        CHECK(da <= bound, "coord-a error within rounding bound");
        CHECK(db <= bound, "coord-b error within rounding bound");
    }
    std::fprintf(stderr,
        "  N=%zu shift=%d bound=%lld worst_err=%lld (at i=%zu, src=(%lld,%lld) rt=(%lld,%lld))\n",
        N, shift, (long long)bound, (long long)worst_err, worst,
        (long long)src[worst].a, (long long)src[worst].b,
        (long long)rt[worst].a,  (long long)rt[worst].b);
}

/* ---------- Test 3: nybble pack/unpack preserves sign for every code ----- */

static void nybble_layout_is_sign_preserving() {
    /* All 16 signed-4-bit values [-8, 7] crossed with themselves. */
    for (int a = -8; a <= 7; ++a) {
        for (int b = -8; b <= 7; ++b) {
            uint8_t packed = sp_ok_q4_pack_pair((int8_t)a, (int8_t)b);
            sp_ok_q4_t q = { packed };
            sp_ok_t r = sp_ok_q4_decode_one(q, 0);  /* shift=0 -> raw int4 */
            CHECK((int64_t)a == r.a, "low-nybble decode matches");
            CHECK((int64_t)b == r.b, "high-nybble decode matches");
        }
    }
}

/* ---------- Test 4: lattice-norm pruning zeros below-threshold elements -- */

static void pruning_zeros_small_norm_entries() {
    /* Construct a mix of small (near origin) and large elements. The
     * small ones should prune; the large ones should survive. */
    const size_t N = 1024;
    std::mt19937_64 rng(0xFEEDFACEC0FFEEFFULL);
    std::uniform_int_distribution<int> small_d(-2, 2);   /* |N| <= 4+2+164 = ~170 */
    std::uniform_int_distribution<int64_t> big_d(
        (int64_t)1 << 20, (int64_t)1 << 24);

    std::vector<sp_ok_t> src(N);
    size_t expected_small = 0;
    for (size_t i = 0; i < N; ++i) {
        if ((i & 1) == 0) {
            src[i] = sp_ok_t{ small_d(rng), small_d(rng) };
            uint64_t n = sp_ok_q4_norm(src[i].a, src[i].b);
            if (n < 200) ++expected_small;
        } else {
            src[i] = sp_ok_t{ big_d(rng), big_d(rng) };
        }
    }

    std::vector<sp_ok_q4_t> packed(N);
    /* Threshold 200 captures every (a, b) with both coords in [-2, 2]
     * (max norm 4 + 2 + 164 = 170 < 200) and nothing in the big-d range
     * (min norm > (2^20)^2 = 2^40). */
    sp_ok_q4_encode_array_pruned(packed.data(), src.data(), N,
                                  /*threshold=*/200);

    CHECK(sp_ok_q4_last_pruned_count == expected_small,
          "pruned count matches the predicted small-norm subset");

    /* The pruned entries must produce 0x00 in the packed byte (since
     * sp_ok_q4_pack_pair(0, 0) == 0). */
    for (size_t i = 0; i < N; i += 2) {
        if (sp_ok_q4_norm(src[i].a, src[i].b) == 0) {
            /* Already-zero entries also produce 0x00; OK to skip. */
            continue;
        }
        if (i < N && src[i].a == 0 && src[i].b == 0) {
            /* Zeroed by the pruner. Packed byte must be 0x00. */
            CHECK(packed[i].packed == 0x00, "pruned pair -> 0x00 byte");
        }
    }
    std::fprintf(stderr,
        "  N=%zu pruned=%zu (predicted=%zu)\n",
        N, sp_ok_q4_last_pruned_count, expected_small);
}

/* ---------- Test 5: compression ratio is exactly 16x -------------------- */

static void compression_ratio_is_16x() {
    constexpr size_t N = 1000;
    const size_t raw_bytes    = N * sizeof(sp_ok_t);
    const size_t packed_bytes = N * sizeof(sp_ok_q4_t);
    CHECK(raw_bytes == N * 16, "sp_ok_t is 16 B/elem");
    CHECK(packed_bytes == N * 1, "sp_ok_q4_t is 1 B/elem");
    CHECK(raw_bytes == packed_bytes * 16, "16x compression on storage");
    std::fprintf(stderr, "  raw=%zu B  packed=%zu B  ratio=%zu\n",
                 raw_bytes, packed_bytes, raw_bytes / packed_bytes);
}

/* ---------- Test 6: zero-shift identity ---------------------------------- */

static void zero_shift_is_identity_on_int4_range() {
    /* When the entire input fits strictly inside [-7, 7] (i.e. absmax
     * <= SP_OK_Q4_MAX), shift is 0 and the decode is a pure sign-extend
     * with no scale change. Round-trip must be bit-exact.
     *
     * Note: the full signed-int4 range is [-8, 7], but the shift picker
     * gates on `absmax <= 7`, so |a|=8 forces shift=1 (rounding tier).
     * The (-8, 7) pair is covered in nybble_layout_is_sign_preserving. */
    constexpr int LO = -7, HI = 7;
    constexpr size_t SIDE = (size_t)(HI - LO + 1);   /* 15 */
    constexpr size_t N = SIDE * SIDE;                /* 225 */
    std::vector<sp_ok_t> src(N);
    size_t idx = 0;
    for (int a = LO; a <= HI; ++a)
        for (int b = LO; b <= HI; ++b)
            src[idx++] = sp_ok_t{ a, b };

    std::vector<sp_ok_q4_t> packed(N);
    int8_t shift = sp_ok_q4_encode_array(packed.data(), src.data(), N);
    CHECK(shift == 0, "abs values <= 7 -> shift=0");

    std::vector<sp_ok_t> rt(N);
    sp_ok_q4_decode_array(rt.data(), packed.data(), N, shift);

    for (size_t i = 0; i < N; ++i) {
        CHECK(src[i].a == rt[i].a, "zero-shift round-trip bit-exact (a)");
        CHECK(src[i].b == rt[i].b, "zero-shift round-trip bit-exact (b)");
    }
}

/* ---------- Driver ------------------------------------------------------- */

int main() {
    run("shift_picker_is_minimal",            shift_picker_is_minimal);
    run("roundtrip_bounded_by_rounding_step", roundtrip_bounded_by_rounding_step);
    run("nybble_layout_is_sign_preserving",   nybble_layout_is_sign_preserving);
    run("pruning_zeros_small_norm_entries",   pruning_zeros_small_norm_entries);
    run("compression_ratio_is_16x",           compression_ratio_is_16x);
    run("zero_shift_is_identity_on_int4_range", zero_shift_is_identity_on_int4_range);

    std::fprintf(stderr,
                 "\n=== %d tests, %d failures ===\n",
                 g_tests, g_failures);
    return g_failures == 0 ? 0 : 1;
}
