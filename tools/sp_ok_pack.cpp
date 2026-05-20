/* sp_ok_pack — Strike 2 v2: Offline GGUF → .sp_ok compiler.
 *
 * Reads a GGUF model and emits a flat binary file laid out as a header
 * + tensor directory + page-aligned blob of sp_ok_q8_block_t structs.
 * The Frobenius (B_a, B_b) coefficients are pre-fused per the math in
 * sp_ok_block_q8_from_gguf_q8_0, so the on-device loader does ZERO
 * math — just pread directly into ION/DMA-BUF pages and the bytes
 * are HTP-ready.
 *
 * File format:
 *
 *   [0,    64)    sp_ok_file_header — magic, version, k, p, scale_recip
 *   [64,   D)     tensor directory — N × sp_ok_tensor_entry (64 B each)
 *   [P0,   ...)   blob, each tensor's blocks aligned to a 4 KB page
 *
 * Where P0 is the first 4 KB boundary after the directory ends.
 *
 * The .sp_ok file is bit-identical to what the desktop engine produces
 * in memory via sp_ok_block_q8_from_gguf_q8_0 — same code path, same
 * math, no divergence. Both desktop and phone consume the same blob.
 *
 * Q8_0 only in v1. Q4_0 / Q4_1 / Q4_K / Q5_0 fanout: extend the type
 * enum and add per-type importer calls below — the directory entry
 * stays the same shape.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#include "ggml.h"
#include "gguf.h"

extern "C" {
#include "sp_ok_block_quant.h"
}

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define SP_OK_MAGIC          0x4B4F5053u   /* "SPOK" LE */
#define SP_OK_VERSION        1u
#define SP_OK_TYPE_Q8_BLOCK  1u
#define SP_OK_PAGE           4096u

#pragma pack(push, 1)
struct sp_ok_file_header {
    uint32_t magic;          /* SP_OK_MAGIC */
    uint32_t version;        /* SP_OK_VERSION */
    uint32_t k_factor;       /* Frobenius power, e.g. 2 */
    uint32_t prime_p;        /* split prime, e.g. 41 */
    uint32_t num_tensors;    /* count of directory entries */
    uint64_t scale_recip;    /* engine production: 16384 = 2^14 */
    uint8_t  reserved[36];   /* pad to 64 */
};
struct sp_ok_tensor_entry {
    char     name[32];       /* null-terminated; longer names get truncated */
    uint32_t type;           /* SP_OK_TYPE_Q8_BLOCK (room for Q4 etc.) */
    uint32_t num_elements;   /* total fp32 elements in the original tensor */
    uint64_t file_offset;    /* page-aligned start of this tensor's blocks */
    uint64_t byte_size;      /* total bytes occupied by sp_ok_q8_block_t array */
    uint8_t  reserved[8];
};
#pragma pack(pop)

static_assert(sizeof(sp_ok_file_header) == 64, "header must be 64 B");
static_assert(sizeof(sp_ok_tensor_entry) == 64, "tensor entry must be 64 B");

static long pad_to_page(FILE* f) {
    long pos = ftell(f);
    long need = (long)(SP_OK_PAGE - (pos % SP_OK_PAGE)) % (long)SP_OK_PAGE;
    if (need > 0) {
        std::vector<uint8_t> zeros((size_t)need, 0);
        std::fwrite(zeros.data(), 1, (size_t)need, f);
    }
    return ftell(f);
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <input.gguf> <output.sp_ok> [options]\n"
        "options:\n"
        "  --k N            Frobenius power (default: 2)\n"
        "  --p N            split prime     (default: 41)\n"
        "  --scale-recip N  Q-format scale  (default: 16384)\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    int     k_factor    = 2;
    int     prime_p     = 41;
    int64_t scale_recip = 16384;
    for (int i = 3; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--k") && i + 1 < argc) k_factor    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--p") && i + 1 < argc) prime_p     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--scale-recip") && i + 1 < argc) scale_recip = std::atoll(argv[++i]);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    std::fprintf(stderr, "sp_ok_pack: %s -> %s  (p=%d k=%d scale_recip=%lld)\n",
                 in_path, out_path, prime_p, k_factor, (long long)scale_recip);

    /* Open GGUF — ask ggml to allocate so t->data points at real bytes. */
    ggml_context*    ggml_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = false;
    gip.ctx      = &ggml_ctx;
    gguf_context* gg = gguf_init_from_file(in_path, gip);
    if (!gg || !ggml_ctx) {
        std::fprintf(stderr, "ERROR: gguf_init_from_file(%s) failed\n", in_path);
        return 1;
    }

    /* First pass: select Q8_0 tensors. */
    int64_t n_tensors = gguf_get_n_tensors(gg);
    struct Selected { int64_t gguf_idx; const char* name; ggml_tensor* t; };
    std::vector<Selected> sel;
    int skipped_type = 0, skipped_shape = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gg, i);
        ggml_tensor* t   = ggml_get_tensor(ggml_ctx, name);
        if (!t) { ++skipped_shape; continue; }
        if (t->type != GGML_TYPE_Q8_0) { ++skipped_type; continue; }
        const int64_t numel = ggml_nelements(t);
        if (numel % SP_OK_BLOCK_SIZE != 0) {
            std::fprintf(stderr, "  skip %s: numel=%lld not multiple of 32\n",
                         name, (long long)numel);
            ++skipped_shape;
            continue;
        }
        sel.push_back({ i, name, t });
    }
    std::fprintf(stderr, "  GGUF tensors total: %lld\n", (long long)n_tensors);
    std::fprintf(stderr, "  Q8_0 selected:      %zu\n", sel.size());
    std::fprintf(stderr, "  skipped (non-Q8):   %d\n", skipped_type);
    std::fprintf(stderr, "  skipped (shape):    %d\n", skipped_shape);
    if (sel.empty()) {
        std::fprintf(stderr, "ERROR: nothing to pack (no Q8_0 tensors)\n");
        ggml_free(ggml_ctx);
        gguf_free(gg);
        return 1;
    }

    /* Open output. */
    FILE* fout = std::fopen(out_path, "wb");
    if (!fout) {
        std::fprintf(stderr, "ERROR: fopen(%s) for write failed\n", out_path);
        ggml_free(ggml_ctx);
        gguf_free(gg);
        return 1;
    }

    /* Write header. */
    sp_ok_file_header hdr{};
    hdr.magic       = SP_OK_MAGIC;
    hdr.version     = SP_OK_VERSION;
    hdr.k_factor    = (uint32_t)k_factor;
    hdr.prime_p     = (uint32_t)prime_p;
    hdr.num_tensors = (uint32_t)sel.size();
    hdr.scale_recip = (uint64_t)scale_recip;
    std::fwrite(&hdr, sizeof(hdr), 1, fout);

    /* Reserve space for the directory; we'll backfill after the blob is written. */
    const long directory_pos = std::ftell(fout);
    std::vector<sp_ok_tensor_entry> directory(sel.size(), sp_ok_tensor_entry{});
    std::fwrite(directory.data(), sizeof(sp_ok_tensor_entry), directory.size(), fout);
    pad_to_page(fout);

    /* Walk selected tensors: fuse each, write blob, fill directory entry. */
    size_t total_blocks  = 0;
    size_t total_bytes   = 0;
    int    failed_import = 0;
    for (size_t idx = 0; idx < sel.size(); ++idx) {
        const auto& s = sel[idx];
        const int64_t numel    = ggml_nelements(s.t);
        const size_t  n_blocks = (size_t)(numel / SP_OK_BLOCK_SIZE);

        std::vector<sp_ok_q8_block_t> fused(n_blocks);
        sp_ok_block_q8_tensor dst{};
        dst.blocks      = fused.data();
        dst.numel       = (size_t)numel;
        dst.n_blocks    = n_blocks;
        dst.frobenius_p = (int16_t)prime_p;
        dst.frobenius_k = (int16_t)k_factor;

        const sp_gguf_block_q8_0* src = (const sp_gguf_block_q8_0*)s.t->data;
        if (!sp_ok_block_q8_from_gguf_q8_0(&dst, src, n_blocks,
                                            scale_recip, prime_p, k_factor)) {
            std::fprintf(stderr, "  ERROR: import failed for %s\n", s.name);
            ++failed_import;
            continue;
        }

        const long file_offset = std::ftell(fout);
        const size_t bytes     = n_blocks * sizeof(sp_ok_q8_block_t);
        std::fwrite(fused.data(), 1, bytes, fout);
        pad_to_page(fout);

        auto& entry = directory[idx];
        std::memset(entry.name, 0, sizeof(entry.name));
        std::strncpy(entry.name, s.name, sizeof(entry.name) - 1);
        if (std::strlen(s.name) >= sizeof(entry.name)) {
            std::fprintf(stderr, "  WARN: name truncated: '%s' -> '%s'\n",
                         s.name, entry.name);
        }
        entry.type         = SP_OK_TYPE_Q8_BLOCK;
        entry.num_elements = (uint32_t)numel;
        entry.file_offset  = (uint64_t)file_offset;
        entry.byte_size    = (uint64_t)bytes;

        total_blocks += n_blocks;
        total_bytes  += bytes;

        if ((idx & 31) == 0 || idx == sel.size() - 1) {
            std::fprintf(stderr, "  [%zu/%zu] %-32s %zu blocks @ off %lld\n",
                         idx + 1, sel.size(), s.name, n_blocks, (long long)file_offset);
        }
    }

    /* Backfill the directory with real offsets/sizes. */
    std::fseek(fout, directory_pos, SEEK_SET);
    std::fwrite(directory.data(), sizeof(sp_ok_tensor_entry), directory.size(), fout);
    std::fseek(fout, 0, SEEK_END);
    const long final_size = std::ftell(fout);
    std::fclose(fout);

    std::fprintf(stderr, "\n=== sp_ok_pack complete ===\n");
    std::fprintf(stderr, "tensors written: %zu\n", sel.size() - failed_import);
    std::fprintf(stderr, "import failures: %d\n", failed_import);
    std::fprintf(stderr, "total blocks:    %zu\n", total_blocks);
    std::fprintf(stderr, "blob bytes:      %.2f MiB\n", total_bytes / (1024.0 * 1024.0));
    std::fprintf(stderr, "file size:       %.2f MiB\n", final_size / (1024.0 * 1024.0));
    std::fprintf(stderr, "Frobenius:       p=%d k=%d scale_recip=%lld\n",
                 prime_p, k_factor, (long long)scale_recip);

    ggml_free(ggml_ctx);
    gguf_free(gg);
    return (failed_import == 0) ? 0 : 2;
}
