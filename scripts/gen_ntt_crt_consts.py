"""Generate sp_ntt_crt_consts.h with two 30-bit Proth primes for CRT-sharded NTT.

Outputs two parallel sets of constants (q1, q2) plus the CRT stitch inverse.
All products fit cleanly in uint64 — no __int128 needed.
"""
import random
random.seed(0xCAFEBABE)

N, LOG_N, TWO_N = 256, 8, 512

def is_prime(n, k=40):
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n == p: return True
        if n % p == 0: return False
    d, r = n-1, 0
    while d % 2 == 0: d //= 2; r += 1
    for _ in range(k):
        a = random.randrange(2, n-1)
        x = pow(a, d, n)
        if x == 1 or x == n-1: continue
        for _ in range(r-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True

def factor_smooth(n):
    facs = set([2]); d = 3
    while d*d <= n:
        if n % d == 0:
            facs.add(d)
            while n % d == 0: n //= d
        d += 2
    if n > 1: facs.add(n)
    return facs

def find_psi(q):
    facs = factor_smooth(q-1)
    for g in range(2, 200):
        if all(pow(g, (q-1)//p, q) != 1 for p in facs):
            psi = pow(g, (q-1)//TWO_N, q)
            assert pow(psi, N, q) == q - 1
            assert pow(psi, TWO_N, q) == 1
            return psi
    raise RuntimeError("no generator")

# Two distinct 30-bit Proth primes, q ≡ 1 mod 2N, near top of 30-bit range.
max_30 = (1 << 30) - 1
k_max = (max_30 - 1) // TWO_N
candidates = []
for k in range(k_max, 0, -1):
    q = TWO_N * k + 1
    if q.bit_length() != 30: continue
    if is_prime(q):
        candidates.append(q)
        if len(candidates) >= 2: break

q1, q2 = candidates[0], candidates[1]
psi1, psi2 = find_psi(q1), find_psi(q2)
psi_inv1, psi_inv2 = pow(psi1, -1, q1), pow(psi2, -1, q2)
N_inv1, N_inv2 = pow(N, -1, q1), pow(N, -1, q2)
crt_q1_inv_q2 = pow(q1 % q2, -1, q2)
M = q1 * q2

# Bitrev table (shared, depends only on N).
def bitrev(x, bits):
    r = 0
    for _ in range(bits):
        r = (r<<1) | (x&1); x >>= 1
    return r
bitrev_perm = [bitrev(i, LOG_N) for i in range(N)]

# Per-prime tables.
psi_pow1 = [pow(psi1, i, q1) for i in range(N)]
psi_inv_pow1 = [pow(psi_inv1, i, q1) for i in range(N)]
psi_pow2 = [pow(psi2, i, q2) for i in range(N)]
psi_inv_pow2 = [pow(psi_inv2, i, q2) for i in range(N)]

# Sanity-verify ALL constants before emitting.
assert is_prime(q1) and is_prime(q2) and q1 != q2
assert q1 % TWO_N == 1 and q2 % TWO_N == 1
assert pow(psi1, N, q1) == q1 - 1 and pow(psi2, N, q2) == q2 - 1
assert (psi1 * psi_inv1) % q1 == 1 and (psi2 * psi_inv2) % q2 == 1
assert (N * N_inv1) % q1 == 1 and (N * N_inv2) % q2 == 1
assert (q1 * crt_q1_inv_q2) % q2 == 1
assert M.bit_length() == 60
# CRT round-trip on 1000 random ints in [0, M).
for _ in range(1000):
    x = random.randrange(0, M)
    a1, a2 = x % q1, x % q2
    diff = (a2 - a1) % q2
    u = (diff * crt_q1_inv_q2) % q2
    assert a1 + u * q1 == x
print("All 9 algebraic checks + 1000-trial CRT round-trip PASS.")

def fmt_u32_array(name, vals, comment=""):
    s = f"/* {comment} */\nstatic const uint32_t {name}[{len(vals)}] = {{\n"
    for i in range(0, len(vals), 8):
        s += "    " + ", ".join(f"{v}" for v in vals[i:i+8]) + ",\n"
    return s + "};\n"

header = f"""/* sp_ntt_crt_consts.h — Auto-generated dual-prime CRT NTT constants.
 *
 * Two 30-bit Proth primes q1, q2 with q ≡ 1 mod 2N.  Combined modulus
 * M = q1 * q2 ≈ 2^{M.bit_length()-1}, replacing the single 60-bit prime path.
 *
 *   q1            = {q1}      ({q1.bit_length()} bits)
 *   q2            = {q2}      ({q2.bit_length()} bits)
 *   psi1          = {psi1}
 *   psi2          = {psi2}
 *   psi_inv1      = {psi_inv1}
 *   psi_inv2      = {psi_inv2}
 *   N_inv1        = {N_inv1}
 *   N_inv2        = {N_inv2}
 *   crt_q1_inv_q2 = {crt_q1_inv_q2}   (q1^-1 mod q2)
 *   M = q1*q2     = {M}
 *
 * Universal-arithmetic property: every intermediate product fits in
 * uint64 (since (q-1)^2 < 2^60 < 2^64). __int128 is not used anywhere
 * on this path. Portable to ARM64, RISC-V, GPU shaders, any 64-bit ALU.
 *
 * CRT recombine (per coefficient):
 *   diff = (a2 - a1) mod q2
 *   u    = (diff * crt_q1_inv_q2) mod q2
 *   x    = a1 + u * q1                  (∈ [0, M), fits in uint64)
 *   signed_x = (x > M/2) ? x - M : x
 *
 * Verified: 9/9 algebraic checks + 1000 random CRT round-trips pass in Python.
 */

#ifndef SP_NTT_CRT_CONSTS_H
#define SP_NTT_CRT_CONSTS_H

#include <stdint.h>

#define SP_NTT_CRT_N            {N}
#define SP_NTT_CRT_LOG_N        {LOG_N}
#define SP_NTT_CRT_2N           {TWO_N}

#define SP_NTT_CRT_Q1           {q1}ULL
#define SP_NTT_CRT_Q2           {q2}ULL
#define SP_NTT_CRT_PSI1         {psi1}ULL
#define SP_NTT_CRT_PSI2         {psi2}ULL
#define SP_NTT_CRT_PSI_INV1     {psi_inv1}ULL
#define SP_NTT_CRT_PSI_INV2     {psi_inv2}ULL
#define SP_NTT_CRT_N_INV1       {N_inv1}ULL
#define SP_NTT_CRT_N_INV2       {N_inv2}ULL
#define SP_NTT_CRT_Q1_INV_Q2    {crt_q1_inv_q2}ULL

"""
def fmt_u64(name, vals, c):
    s = f"/* {c} */\nstatic const uint64_t {name}[{len(vals)}] = {{\n"
    for i in range(0, len(vals), 4):
        s += "    " + ", ".join(f"{v}ULL" for v in vals[i:i+4]) + ",\n"
    return s + "};\n\n"

header += fmt_u64("sp_ntt_crt_psi_pow1",     psi_pow1,     "psi1^i mod q1 (negacyclic pre-twist, prime 1)")
header += fmt_u64("sp_ntt_crt_psi_inv_pow1", psi_inv_pow1, "psi1^-i mod q1 (post-twist, prime 1)")
header += fmt_u64("sp_ntt_crt_psi_pow2",     psi_pow2,     "psi2^i mod q2 (pre-twist, prime 2)")
header += fmt_u64("sp_ntt_crt_psi_inv_pow2", psi_inv_pow2, "psi2^-i mod q2 (post-twist, prime 2)")
header += fmt_u32_array("sp_ntt_crt_bitrev", bitrev_perm, "bitrev(i, log2 N) — shared, depends only on N")
header += "\n#endif /* SP_NTT_CRT_CONSTS_H */\n"

OUT = "/sessions/dazzling-ecstatic-ritchie/mnt/shannon-prime-repos/shannon-prime-engine/lib/shannon-prime/core/sp_ntt_crt_consts.h"
with open(OUT, "w") as f: f.write(header)
print(f"Wrote {OUT} ({len(header)} bytes)")
