// Shannon-Prime Engine — Frobenius / Sato-Tate quant dispatch
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// C++ wrapper around the C-side sp_frobenius routines. Provides the
// engine-level entry points used by sp_quant.cpp and the CLI.

#include "sp_quant_frobenius.h"

extern "C" {
#include "sp_ok_arith.h"
#include "sp_frobenius.h"
}

#include <cstdio>
#include <vector>

namespace sp::engine {

void apply_frobenius_quant_inplace(sp_ok_t *state, size_t n_elements,
                                    int64_t p, int64_t k) {
    sp_frobenius_quant_tensor(state, n_elements, p, k);
}

void apply_sato_tate_mix_inplace(sp_ok_t *state, size_t n_elements,
                                  int64_t p1, int64_t k1,
                                  int64_t p2, int64_t k2) {
    sp_sato_tate_mix_tensor(state, n_elements, p1, k1, p2, k2);
}

bool parse_sato_tate_mix_args(const std::string &arg,
                               int64_t &p1, int64_t &k1,
                               int64_t &p2, int64_t &k2) {
    // Format: "p1,k1,p2,k2"
    long long a, b, c, d;
    if (std::sscanf(arg.c_str(), "%lld,%lld,%lld,%lld", &a, &b, &c, &d) != 4) {
        return false;
    }
    p1 = (int64_t)a; k1 = (int64_t)b;
    p2 = (int64_t)c; k2 = (int64_t)d;
    if (!sp_is_inert(p1) && !sp_is_split(p1)) return false;
    if (!sp_is_split(p2) && !sp_is_inert(p2)) return false;
    return true;
}

void print_frobenius_diagnostics(int64_t p, int64_t k) {
    std::printf("[sp-frobenius] tier: p=%lld, k=%lld\n", (long long)p, (long long)k);
    if (sp_is_inert(p)) {
        std::printf("[sp-frobenius]   p is INERT in K=Q(sqrt(-163)); phi_p^2 = -p (zero drift)\n");
    } else if (sp_is_split(p)) {
        sp_ok_t pi;
        if (sp_find_element_of_norm(p, &pi)) {
            std::printf("[sp-frobenius]   p is SPLIT; pi = (%lld, %lld), trace = %lld\n",
                (long long)pi.a, (long long)pi.b, (long long)sp_ok_trace(pi));
        }
    } else if (sp_is_ramified(p)) {
        std::printf("[sp-frobenius]   p is RAMIFIED (unsupported)\n");
    }
}

void print_sato_tate_diagnostics(int64_t p1, int64_t k1, int64_t p2, int64_t k2) {
    std::printf("[sp-sato-tate-mix] inert channel:  p1=%lld k1=%lld (",
        (long long)p1, (long long)k1);
    std::printf(sp_is_inert(p1) ? "INERT" : "split");
    std::printf(")\n");
    std::printf("[sp-sato-tate-mix] split channel:  p2=%lld k2=%lld (",
        (long long)p2, (long long)k2);
    std::printf(sp_is_split(p2) ? "SPLIT" : "inert");
    std::printf(")\n");
    sp_ok_t pi;
    if (sp_is_split(p2) && sp_find_element_of_norm(p2, &pi)) {
        int64_t ap = sp_ok_trace(pi);
        std::printf("[sp-sato-tate-mix]   a_p2 = %lld; drift bound = %lld\n",
            (long long)ap, (long long)k2 * (ap < 0 ? -ap : ap));
    }
}

} // namespace sp::engine
