"""Generate sp_ntt_consts.h matching the clean textbook NTT layout that passed
the 5/5 self-check in /tmp/clean_ntt.py."""
import random, math, os, shutil
random.seed(0xC0DEFACE)

N = 256
LOG_N = 8
TWO_N = 512

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

SHIFT = 16
target_min = 1 << 59
k_start = (target_min - 1 + (1 << SHIFT) - 1) // (1 << SHIFT)
if k_start % 2 == 0: k_start += 1
k = k_start
while True:
    cand = k * (1 << SHIFT) + 1
    if is_prime(cand):
        q = cand; break
    k += 2
assert q % TWO_N == 1

def factor(n):
    facs = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            facs[d] = facs.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1: facs[n] = facs.get(n, 0) + 1
    return facs

prime_div = set([2]) | set(factor((q - 1) >> SHIFT).keys())

def is_primitive(g):
    for p in prime_div:
        if pow(g, (q-1)//p, q) == 1: return False
    return True

g = 2
while not is_primitive(g): g += 1

psi = pow(g, (q-1)//TWO_N, q)
psi_inv = pow(psi, -1, q)
omega = pow(psi, 2, q)
omega_inv = pow(omega, -1, q)
N_inv = pow(N, -1, q)
assert pow(psi, N, q) == q - 1
assert pow(omega, N, q) == 1

print(f"q       = {q}  ({q.bit_length()} bits)")
print(f"psi     = {psi}")
print(f"psi_inv = {psi_inv}")
print(f"omega   = {omega}")
print(f"N_inv   = {N_inv}")

# Tables
psi_pow = [pow(psi, i, q) for i in range(N)]
psi_inv_pow = [pow(psi_inv, i, q) for i in range(N)]

# Bit-reverse permutation table
def bitrev(x, bits):
    r = 0
    for _ in range(bits):
        r = (r<<1) | (x&1); x >>= 1
    return r
bitrev_perm = [bitrev(i, LOG_N) for i in range(N)]

def fmt_u64_array(name, vals, comment=""):
    s = f"/* {comment} */\n" if comment else ""
    s += f"static const uint64_t {name}[{len(vals)}] = {{\n"
    for i in range(0, len(vals), 4):
        chunk = vals[i:i+4]
        s += "    " + ", ".join(f"{v}ULL" for v in chunk) + ",\n"
    s += "};\n"
    return s

def fmt_u32_array(name, vals, comment=""):
    s = f"/* {comment} */\n" if comment else ""
    s += f"static const uint32_t {name}[{len(vals)}] = {{\n"
    for i in range(0, len(vals), 16):
        chunk = vals[i:i+16]
        s += "    " + ", ".join(f"{v}" for v in chunk) + ",\n"
    s += "};\n"
    return s

header = f"""/*
 * sp_ntt_consts.h — Auto-generated NTT constants for negacyclic polynomial
 *                   multiplication in Z_q[x]/(x^N + 1), N = {N}.
 *
 * Generator: shannon-prime-engine/scripts/gen_ntt_consts.py
 *
 *   q         = {q}        ({q.bit_length()}-bit Proth prime)
 *   N         = {N}                                  (transform length, log2 N = {LOG_N})
 *   psi       = {psi}    (primitive 2N-th root of unity, psi^N ≡ -1 mod q)
 *   psi_inv   = {psi_inv}
 *   omega     = psi^2 = {omega}
 *   omega_inv = {omega_inv}
 *   N_inv     = {N_inv}      (N^-1 mod q)
 *
 * Forward negacyclic NTT pipeline (clean textbook form):
 *
 *   1. Pre-twist:        a'_i = a_i * sp_ntt_psi_pow[i]  mod q
 *   2. Bit-reverse:      a'   = bitrev_permute(a')
 *   3. Cyclic NTT:       Cooley-Tukey radix-2 DIT, natural-order twiddles
 *                        w_step at layer length L is omega^(N/L);
 *                        w iterates 1, w_step, w_step^2, ... within each block.
 *
 * Inverse negacyclic NTT:
 *
 *   1. Bit-reverse:      A   = bitrev_permute(A)
 *   2. Cyclic inv NTT:   same butterflies, w_step = omega_inv^(N/L), then multiply by N_inv
 *   3. Post-twist:       a_i = A_i * sp_ntt_psi_inv_pow[i] mod q
 *
 * Self-check (Python reference, /tmp/clean_ntt.py): 5/5 PASS
 *   - roundtrip, random poly mul, identity mul, x*x^(N-1)=-1, dot-product use case
 */

#ifndef SP_NTT_CONSTS_H
#define SP_NTT_CONSTS_H

#include <stdint.h>

#define SP_NTT_Q         {q}ULL
#define SP_NTT_N         {N}
#define SP_NTT_LOG_N     {LOG_N}
#define SP_NTT_2N        {TWO_N}
#define SP_NTT_PSI       {psi}ULL
#define SP_NTT_PSI_INV   {psi_inv}ULL
#define SP_NTT_OMEGA     {omega}ULL
#define SP_NTT_OMEGA_INV {omega_inv}ULL
#define SP_NTT_N_INV     {N_inv}ULL

"""
header += fmt_u64_array("sp_ntt_psi_pow", psi_pow, "psi^i mod q  (negacyclic pre-twist)")
header += "\n"
header += fmt_u64_array("sp_ntt_psi_inv_pow", psi_inv_pow, "psi^-i mod q  (negacyclic post-twist)")
header += "\n"
header += fmt_u32_array("sp_ntt_bitrev", bitrev_perm, "bitrev(i, log2 N) — bit-reverse permutation index table")
header += "\n#endif /* SP_NTT_CONSTS_H */\n"

OUT = "/sessions/dazzling-ecstatic-ritchie/mnt/shannon-prime-repos/shannon-prime-engine/lib/shannon-prime/core/sp_ntt_consts.h"
with open(OUT, "w") as f:
    f.write(header)
print(f"\nWrote: {OUT} ({len(header)} bytes)")

GEN = "/sessions/dazzling-ecstatic-ritchie/mnt/shannon-prime-repos/shannon-prime-engine/scripts/gen_ntt_consts.py"
os.makedirs(os.path.dirname(GEN), exist_ok=True)
shutil.copy("/tmp/gen_ntt_final.py", GEN)
print(f"Wrote: {GEN}")
