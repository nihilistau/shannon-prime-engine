// Shannon-Prime Engine — Frobenius / Sato-Tate quant dispatch
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Engine-facing entry points to the C-side sp_frobenius routines.

#pragma once

#include "../lib/shannon-prime/core/sp_ok_arith.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace sp::engine {

// Apply phi_p^k to every state element of an O_K-coordinate tensor.
// state is overwritten in place. Config B (Paper D §2.3): p=41 split, k=8.
void apply_frobenius_quant_inplace(sp_ok_t *state, std::size_t n_elements,
                                    int64_t p, int64_t k);

// Apply the composite phi_p1^k1 ∘ phi_p2^k2 (Sato-Tate mixed precision).
// Config E (Paper D §2.3): p1=2 inert, k1=2; p2=41 split, k2=8.
void apply_sato_tate_mix_inplace(sp_ok_t *state, std::size_t n_elements,
                                  int64_t p1, int64_t k1,
                                  int64_t p2, int64_t k2);

// Parse a CLI arg of the form "p1,k1,p2,k2". Returns true on success.
bool parse_sato_tate_mix_args(const std::string &arg,
                               int64_t &p1, int64_t &k1,
                               int64_t &p2, int64_t &k2);

// Engine-startup diagnostics: print classification of chosen primes.
void print_frobenius_diagnostics(int64_t p, int64_t k);
void print_sato_tate_diagnostics(int64_t p1, int64_t k1,
                                  int64_t p2, int64_t k2);

} // namespace sp::engine
