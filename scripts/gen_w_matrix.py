"""Generate sp_hex_w_matrix_hd<head_dim>.h — compile-time W-matrix for the
Hierarchical Spinor predictor (Strike 11).

The Hierarchical path replaces the Strike-9 pipeline
    VHT2 -> Mobius scatter -> band_quantize (5/5/4/3)
with
    VHT2(mixed-radix 154) -> skeleton extract -> W * skeleton -> residual quant

The W matrix is shape (predicted_size, skeleton_size) = (140, 14) for the
default 154 = 2*7*11 mixed-radix layout (14 Knight skeleton coords + 140
residual coords).

LAYOUT — COLUMN-MAJOR + Q15-IN-LOW-HALF int32 LANES.

  Storage type: int32_t
  Storage shape: (skeleton_size, predicted_padded) where
                 predicted_padded = ceil(predicted_size / 32) * 32 = 160.
  Lane content:  low 16 bits = signed Q15 W value (sign-extended)
                 high 16 bits = 0
  Pad lanes [predicted_size..predicted_padded): all zero (so HVX MACs
  past the live region accumulate nothing).

WHY int32 LANES INSTEAD OF int16:

The validated Strike 5 HVX intrinsic Q6_Vw_vmpyieacc_VwVwVh computes
  Vacc.w[i] += Vu.w[i] * Vv.h[2i]
i.e. word accumulator += word * EVEN halfword. With a tightly-packed
i16 W array, that picks every-other W value (half the data wasted on
odd halfwords). Pre-padding W into i32 lanes with the value in the
low half (the even halfword of that i32 lane) lets one vmpyieacc
process 32 contiguous W values per call — full lane utilization with
the same intrinsic Strike 5 already proved on V69 silicon.

Cost: 2x .rodata (.h grows from 280 to 640 bytes per column, 8960 total
for 14 columns × 160 lanes). Trivial vs the ~64KB VTCM budget.

ACCESS PATTERN:

    for j in 0..skeleton_size:                          # outer over columns
        s_j_q15 = round(skeleton[j] * 32767)            # int32 scalar
        for chunk in 0..predicted_padded, step 32:      # 5 HVX vectors
            v_w = vmem load 32 int32 from column j     # 128 bytes
            v_acc[chunk] = vmpyieacc(v_acc[chunk],
                                      vsplat(s_j_q15),
                                      v_w)              # +32 i32 lanes

QUANTIZATION — Q15 fixed-point. The V69 lacks native fp16 vector MAC
instructions, but executes 64 i16*i16 -> i32 MACs per cycle. We scale
calibrated fp32 weights by 32767 and round to int16, then pad each
value into a low halfword of an i32 lane. The kernel restores fp32
by dividing the i32 accumulator by 32767^2 (single fp multiply).

CALIBRATION — for now this generator emits a deterministic pseudo-random
distribution scaled to 0.05 * N(0,1) so the parity test has stable
weights across Linux/Windows runs. The real generator drops the SVD-
calibrated weights into the same slot when calibration lands.
"""

import os

import numpy as np


# ----------------------------------------------------------------------------
# Output paths — match the gen_mobius_tables.py convention.
# ----------------------------------------------------------------------------
def repo_hexagon_dir():
    """Resolve <repo>/lib/shannon-prime/backends/hexagon regardless of cwd."""
    this = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(this)
    repo = os.path.dirname(scripts_dir)
    return os.path.join(repo, "lib", "shannon-prime", "backends", "hexagon")


# ----------------------------------------------------------------------------
# Calibration source — placeholder until SVD pipeline lands.
# ----------------------------------------------------------------------------
def calibrated_w_matrix(predicted_size, skeleton_size, seed_offset):
    """Return a fp32 W matrix of shape (predicted_size, skeleton_size).

    For Strike 11 scaffold this is deterministic pseudo-random; the seed
    is keyed off head_dim so each supported config has stable weights.
    Replace this function body with the SVD-calibrated load when the
    calibration pipeline ships.
    """
    rng = np.random.default_rng(seed=42 + seed_offset)
    W = rng.standard_normal((predicted_size, skeleton_size)) * 0.05
    return W.astype(np.float32)


# ----------------------------------------------------------------------------
# Q15 quantize + max-abs reporting for the parity test budget.
# ----------------------------------------------------------------------------
def quantize_q15(W_float):
    """fp32 (predicted, skeleton) -> int16 (predicted, skeleton) in Q15."""
    Q_SCALE = 32767.0
    W_scaled = np.round(W_float * Q_SCALE)
    W_q15 = np.clip(W_scaled, -32768, 32767).astype(np.int16)
    abs_err = np.max(np.abs(W_float - W_q15.astype(np.float32) / Q_SCALE))
    return W_q15, abs_err


# ----------------------------------------------------------------------------
# Build the (skeleton_size, predicted_padded) int32 storage from the Q15 int16
# matrix. Each i16 value lands in the low half of its i32 lane; high half = 0;
# pad lanes (predicted_size..predicted_padded) = 0.
# ----------------------------------------------------------------------------
def pack_padded_int32(W_q15_int16, predicted_size, skeleton_size, padded_size):
    """Column-major int32 buffer with Q15 in low half + zero pad."""
    assert W_q15_int16.shape == (predicted_size, skeleton_size)
    out = np.zeros((skeleton_size, padded_size), dtype=np.int32)
    for j in range(skeleton_size):
        out[j, :predicted_size] = W_q15_int16[:, j].astype(np.int32)
    return out


# ----------------------------------------------------------------------------
# Header emitter.
# ----------------------------------------------------------------------------
def emit_header(head_dim, skeleton_size, predicted_size, predicted_padded,
                W_packed_i32, abs_err, out_dir):
    """Write sp_hex_w_matrix_hd<head_dim>.h with the padded int32 W table."""
    assert W_packed_i32.shape == (skeleton_size, predicted_padded)
    assert W_packed_i32.dtype == np.int32

    fname = os.path.join(out_dir, f"sp_hex_w_matrix_hd{head_dim}.h")
    total_elems = skeleton_size * predicted_padded
    total_bytes = total_elems * 4  # int32

    lines = []
    lines.append(
        f"// Auto-generated by scripts/gen_w_matrix.py — do not hand-edit.")
    lines.append(
        f"// W-matrix for Hierarchical Spinor predictor, head_dim={head_dim}.")
    lines.append(
        f"//   logical shape : ({predicted_size}, {skeleton_size}) "
        f"= (predicted, skeleton)")
    lines.append(
        f"//   storage shape : ({skeleton_size}, {predicted_padded}) "
        f"int32, column-major")
    lines.append(
        f"//   per-lane      : low 16 bits = Q15 W value; high 16 bits = 0")
    lines.append(
        f"//   pad lanes     : {predicted_padded - predicted_size} zeros "
        f"per column (lanes {predicted_size}..{predicted_padded})")
    lines.append(
        f"//   total bytes   : {total_bytes} "
        f"({skeleton_size}*{predicted_padded}*4)")
    lines.append(
        f"//   max abs quant error (fp32 vs Q15): {abs_err:.3e}")
    lines.append("//")
    lines.append("// Access pattern: skeleton element j is one contiguous run")
    lines.append("// of `predicted_padded` int32 starting at offset")
    lines.append("//   j * predicted_padded * sizeof(int32).")
    lines.append("//")
    lines.append("// HVX MAC: Q6_Vw_vmpyieacc_VwVwVh(acc, vsplat(s_q15), v_w).")
    lines.append("// Restore fp32 on output: out_f32 = acc_i32 / (32767 * 32767).")
    lines.append("")
    lines.append("#ifndef SP_HEX_W_MATRIX_HD" + str(head_dim) + "_H")
    lines.append("#define SP_HEX_W_MATRIX_HD" + str(head_dim) + "_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(
        f"#define SP_HEX_W_MATRIX_HD{head_dim}_SKELETON         {skeleton_size}")
    lines.append(
        f"#define SP_HEX_W_MATRIX_HD{head_dim}_PREDICTED        {predicted_size}")
    lines.append(
        f"#define SP_HEX_W_MATRIX_HD{head_dim}_PREDICTED_PAD    {predicted_padded}")
    lines.append("")
    lines.append(
        f"static const int32_t sp_hex_w_matrix_hd{head_dim}[{total_elems}]")
    lines.append("    __attribute__((aligned(128))) = {")

    # Column-major dump.
    for j in range(skeleton_size):
        lines.append(f"    /* skeleton element {j} */")
        col = W_packed_i32[j]
        for chunk_start in range(0, predicted_padded, 16):
            chunk = col[chunk_start:chunk_start + 16]
            lines.append("    " + ", ".join(f"{int(v):6d}" for v in chunk) + ",")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    lines.append("};")
    lines.append("")
    lines.append("#endif  // SP_HEX_W_MATRIX_HD" + str(head_dim) + "_H")
    lines.append("")

    os.makedirs(out_dir, exist_ok=True)
    with open(fname, "w", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"[gen_w_matrix] wrote {fname} ({total_bytes} bytes, "
          f"max quant err {abs_err:.3e})")
    return fname


# ----------------------------------------------------------------------------
# Public entry — sweep all supported configs.
# ----------------------------------------------------------------------------
CONFIGS = [
    # Strike 11c reshape: engine's knight_mask at pad_dim=154 produces
    # 14 skeleton + 60 non-squarefree-residual; ~80 squarefree-but-not-
    # skeleton indices are dropped.  Pad to 64 lanes (2 HVX vectors of
    # 32 i32 each) for the MAC kernel.  W-matrix .rodata shrinks from
    # 8960 B (14 cols * 160 lanes * 4) to 3584 B (14 cols * 64 lanes * 4).
    {
        "head_dim":         154,
        "skeleton_size":    14,
        "predicted_size":   60,
        "predicted_padded": 64,
    },
]


def _legacy_size_assertion(cfg):
    """Strike 11c removed the skeleton+predicted == head_dim invariant.
    The engine's knight_mask drops ~80 squarefree-but-not-skeleton indices,
    so 14 + 60 = 74 < 154 is intentional."""
    _ = cfg
    return True


def main():
    out_dir = repo_hexagon_dir()
    print(f"[gen_w_matrix] target dir: {out_dir}")
    for cfg in CONFIGS:
        hd  = cfg["head_dim"]
        sk  = cfg["skeleton_size"]
        pr  = cfg["predicted_size"]
        pad = cfg["predicted_padded"]
        _legacy_size_assertion(cfg)  # skeleton + predicted != head_dim by design
        assert pad >= pr and pad % 32 == 0, (
            f"predicted_padded ({pad}) must be >= predicted_size ({pr}) "
            f"and divisible by 32")
        W = calibrated_w_matrix(pr, sk, seed_offset=hd)
        W_q15, err = quantize_q15(W)
        W_packed = pack_padded_int32(W_q15, pr, sk, pad)
        emit_header(hd, sk, pr, pad, W_packed, err, out_dir)


if __name__ == "__main__":
    main()
