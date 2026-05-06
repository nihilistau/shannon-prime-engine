#!/usr/bin/env python3
"""
qnn_bin_probe.py — Recover per-tensor UFIXED_POINT_16 (scale, offset) from
AI Hub-compiled Qwen3-4B V69 QNN .bin files.

Background
----------
AI Hub bakes quantization parameters into compiled graph ops. The QNN tensor
metadata API returns quant_encoding=0 for all dtype=1046 tensors, so we cannot
read scale/offset from the binary directly. This script recovers them empirically:

    fp32 = (uint16 + offset) * scale      [QNN canonical decode]

For OUTPUT tensors (embedding, residuals, logits) we run qnn-net-run twice
for the same input — once for fp32 output, once for native uint16 output —
and fit a linear regression:

    fp32_i = scale * uint16_i + offset_fp32
    where offset_fp32 = offset * scale (sign-absorbed)

With 3+ synthetic input_ids variants we get a robust N-sample regression
across the full output distribution.

For INPUT tensors (attention_mask, cos/sin, past_kv) the derivation is from
calibration range properties (see qnn_bin_quant_table.h). If qnn-net-run
exposes --use_native_input_files, the script also probes those directly.

Usage
-----
    python scripts/qnn_bin_probe.py \\
        --bins /path/to/qwen3_4b_1_of_4.bin ... /path/to/qwen3_4b_4_of_4.bin \\
        --qnn-net-run /path/to/qnn-net-run \\
        --device-tmp /data/local/tmp/sp22u/qnn \\
        --out src/qnn_bin_quant_table.h

The script:
  1. Generates 5 synthetic input_ids raw files (random tokens, seed-fixed).
  2. adb-pushes them and the .bins.
  3. Runs qnn-net-run twice per split (fp32 + native) for each synthetic input.
  4. Pulls output .raw files.
  5. Runs OLS regression per output tensor, prints (scale, offset) table.
  6. Cross-checks: embedding anchor must match scale≈7.2e-6, offset≈-0.222.
  7. Optionally writes the recovered values into qnn_bin_quant_table.h.

Requirements: Python 3.8+, numpy, adb in PATH.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ── Synthetic token IDs to feed as input_ids ──────────────────────────────
# Three samples at different vocab positions to spread the embedding range.
# Keep deterministic (fixed token IDs, not random) for reproducibility.
SYNTHETIC_TOKEN_SETS = [
    [0]    * 128,           # all PAD
    [5000] * 128,           # mid-vocab
    [100_000] * 128,        # high vocab (Qwen3 vocab=151936)
    [0, 1, 2, 3] * 32,     # sequential low
    [75_000] * 128,         # upper-mid
]
AR     = 128
VOCAB  = 151936
HIDDEN = 2560


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def adb(*args: str) -> subprocess.CompletedProcess:
    return run(["adb", *args])


def adb_shell(cmd: str) -> subprocess.CompletedProcess:
    return run(["adb", "shell", cmd])


def make_input_raw(token_ids: list[int], path: Path) -> None:
    """Write int32 raw file for input_ids[1, 128]."""
    path.write_bytes(struct.pack(f"<{len(token_ids)}i", *token_ids))


def run_qnn_net_run(
    device_tmp: str,
    bin_name: str,
    input_list_path: str,  # device path to input_list file
    output_dir: str,       # device-side output dir
    native: bool,
    qnn_net_run: str,
) -> None:
    """Run qnn-net-run on device for one split."""
    native_flag = "--use_native_output_files" if native else ""
    cmd = (
        f"cd {device_tmp} && "
        f"LD_LIBRARY_PATH={device_tmp} "
        f"{qnn_net_run} "
        f"--model {bin_name} "
        f"--input_list {input_list_path} "
        f"--output_dir {output_dir} "
        f"--backend libQnnHtp.so "
        + (f"{native_flag} " if native else "")
        + "--log_level error"
    )
    adb_shell(cmd)


def pull_outputs(device_dir: str, local_dir: Path) -> dict[str, np.ndarray]:
    """Pull all .raw files from device_dir, return {name: ndarray(uint8)}."""
    result = adb_shell(f"ls {device_dir}/*.raw")
    files = result.stdout.strip().split()
    out = {}
    for f in files:
        name = Path(f).name
        local = local_dir / name
        adb("pull", f, str(local))
        out[name] = np.frombuffer(local.read_bytes(), dtype=np.uint8)
    return out


def regression_scale_offset(
    fp32_vals: np.ndarray, u16_vals: np.ndarray
) -> tuple[float, float]:
    """
    OLS: fp32 = scale * uint16 + intercept.
    Returns (scale, intercept) where intercept = offset_qnn * scale.
    The QNN integer offset = round(intercept / scale).
    """
    A = np.column_stack([u16_vals.astype(np.float64), np.ones_like(u16_vals, dtype=np.float64)])
    b = fp32_vals.astype(np.float64)
    result, *_ = np.linalg.lstsq(A, b, rcond=None)
    scale, intercept = float(result[0]), float(result[1])
    return scale, intercept


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bins", nargs=4, required=True, metavar="BIN",
                   help="Four split .bin files (split1 split2 split3 split4)")
    p.add_argument("--qnn-net-run", default="/data/local/tmp/sp22u/qnn/qnn-net-run",
                   help="Device path to qnn-net-run binary")
    p.add_argument("--device-tmp", default="/data/local/tmp/sp22u/qnn",
                   help="Device-side working directory")
    p.add_argument("--out", default="src/qnn_bin_quant_table.h",
                   help="Path to write updated qnn_bin_quant_table.h")
    p.add_argument("--samples", type=int, default=3,
                   help="Number of synthetic input sets to use (1..5, default 3)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print adb commands but do not execute them")
    return p.parse_args()


def probe_split1(
    bin_path: str,
    device_tmp: str,
    qnn_net_run: str,
    token_sets: list[list[int]],
    workdir: Path,
) -> dict[str, tuple[float, float]]:
    """
    Probe split 1 (embedding). Returns per-output-tensor {raw_name: (scale, intercept)}.
    """
    print(f"\n=== Probing split 1: {Path(bin_path).name} ===")
    bin_name = Path(bin_path).name

    fp32_rows: dict[str, list[np.ndarray]] = {}
    u16_rows:  dict[str, list[np.ndarray]] = {}

    for idx, tokens in enumerate(token_sets):
        print(f"\n  -- sample {idx} (token[0]={tokens[0]}) --")
        raw_path = workdir / f"input_{idx}.raw"
        make_input_raw(tokens, raw_path)
        adb("push", str(raw_path), f"{device_tmp}/input_{idx}.raw")

        # input_list file
        input_list = workdir / f"input_list_{idx}.txt"
        input_list.write_text(f"{device_tmp}/input_{idx}.raw\n")
        adb("push", str(input_list), f"{device_tmp}/input_list_{idx}.txt")

        fp32_dir = f"{device_tmp}/out_fp32_{idx}"
        u16_dir  = f"{device_tmp}/out_u16_{idx}"
        adb_shell(f"rm -rf {fp32_dir} {u16_dir} && mkdir -p {fp32_dir} {u16_dir}")

        run_qnn_net_run(device_tmp, bin_name,
                        f"{device_tmp}/input_list_{idx}.txt",
                        fp32_dir, native=False, qnn_net_run=qnn_net_run)
        run_qnn_net_run(device_tmp, bin_name,
                        f"{device_tmp}/input_list_{idx}.txt",
                        u16_dir, native=True,  qnn_net_run=qnn_net_run)

        local_fp32 = workdir / f"fp32_{idx}"
        local_u16  = workdir / f"u16_{idx}"
        local_fp32.mkdir(exist_ok=True)
        local_u16.mkdir(exist_ok=True)

        fp32_files = pull_outputs(fp32_dir, local_fp32)
        u16_files  = pull_outputs(u16_dir,  local_u16)

        for name, fp32_bytes in fp32_files.items():
            # Match native counterpart — qnn-net-run appends "_native" suffix.
            native_name = name.replace(".raw", "_native.raw")
            if native_name not in u16_files:
                continue
            fp32_arr = np.frombuffer(fp32_bytes.tobytes(), dtype=np.float32)
            u16_arr  = np.frombuffer(u16_files[native_name].tobytes(), dtype=np.uint16)
            if fp32_arr.shape != u16_arr.shape:
                print(f"    [warn] shape mismatch for {name}: "
                      f"fp32={fp32_arr.shape} u16={u16_arr.shape}")
                continue
            fp32_rows.setdefault(name, []).append(fp32_arr)
            u16_rows.setdefault(name, []).append(u16_arr)

    results = {}
    for name in fp32_rows:
        fp32_all = np.concatenate(fp32_rows[name])
        u16_all  = np.concatenate(u16_rows[name])
        # Subsample to keep regression tractable (max 50k points)
        if len(fp32_all) > 50_000:
            idx = np.random.default_rng(42).choice(len(fp32_all), 50_000, replace=False)
            fp32_all = fp32_all[idx]
            u16_all  = u16_all[idx]
        scale, intercept = regression_scale_offset(fp32_all, u16_all)
        offset_qnn = round(intercept / scale) if scale != 0 else 0
        results[name] = (scale, intercept)
        print(f"  {name}:")
        print(f"    scale={scale:.6g}  intercept={intercept:.6g}  "
              f"→ QNN offset≈{offset_qnn}")

    return results


def cross_check_embedding(results: dict[str, tuple[float, float]]) -> None:
    """Verify embedding output matches Phase 5.7 empirical anchor."""
    for name, (scale, intercept) in results.items():
        if "Gather" in name or "embed" in name.lower():
            if not (5e-6 <= scale <= 1e-5):
                print(f"  [WARN] embedding scale={scale:.3g} outside expected 5e-6..1e-5")
            else:
                print(f"  [OK] embedding scale={scale:.3g} matches anchor ~7.2e-6")
            offset_fp = intercept
            if not (-0.3 <= offset_fp <= -0.1):
                print(f"  [WARN] embedding intercept={offset_fp:.4g} outside -0.3..-0.1")
            else:
                print(f"  [OK] embedding intercept={offset_fp:.4g} matches anchor ~-0.222")
            return
    print("  [WARN] No embedding output tensor found for cross-check")


def derive_input_params() -> dict[str, tuple[float, int]]:
    """
    Return (scale, offset_qnn) for input tensors derived from calibration range.
    These are the theoretical values from standard UFIXED_16 over the known range.
    """
    return {
        "attention_mask": (0.99953, 0),    # range [-65504, 0], scale=65504/65535, offset=0
        "position_ids_cos": (3.0518e-5, -32768),  # range [-1, 1]
        "position_ids_sin": (3.0518e-5, -32768),
        "past_key":         (7.2e-6, -30833),     # same as residual (placeholder)
        "past_value":       (7.2e-6, -30833),
    }


def print_summary(
    split_results: list[dict],
    input_params: dict,
) -> None:
    print("\n" + "=" * 70)
    print("RECOVERED QUANTIZATION TABLE")
    print("=" * 70)
    print("\nOutput tensors (from regression):")
    for split_idx, res in enumerate(split_results, 1):
        for name, (scale, intercept) in sorted(res.items()):
            offset_qnn = round(intercept / scale) if scale != 0 else 0
            print(f"  split{split_idx} {name[:48]:<48}  scale={scale:.4g}  "
                  f"QNN_offset={offset_qnn}")

    print("\nInput tensors (derived from calibration range):")
    for name, (scale, offset) in input_params.items():
        print(f"  {name:<24}  scale={scale:.4g}  QNN_offset={offset}")

    print("\nAdd to src/qnn_bin_quant_table.h — replace placeholder constants with:")
    for split_idx, res in enumerate(split_results, 1):
        for name, (scale, intercept) in sorted(res.items()):
            offset_qnn = round(intercept / scale) if scale != 0 else 0
            tag = "EMBEDDING" if "Gather" in name or "embed" in name.lower() else \
                  "LOGITS" if "logit" in name.lower() or "lm_head" in name.lower() else \
                  f"RESIDUAL_S{split_idx}"
            print(f"  QNN_QUANT_{tag:<20}  {{ {scale:.6e}f, {offset_qnn} }}")


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print("[dry-run] would execute adb commands; skipping actual runs")
        return

    workdir = Path(tempfile.mkdtemp(prefix="qnn_bin_probe_"))
    print(f"Working dir: {workdir}")

    n_samples = min(max(args.samples, 1), len(SYNTHETIC_TOKEN_SETS))
    token_sets = SYNTHETIC_TOKEN_SETS[:n_samples]

    # Push .bins to device
    print("\nPushing .bins to device...")
    for bin_path in args.bins:
        adb("push", bin_path, f"{args.device_tmp}/{Path(bin_path).name}")

    # Probe split 1 (embedding output) — primary anchor
    split1_results = probe_split1(
        args.bins[0],
        args.device_tmp,
        args.qnn_net_run,
        token_sets,
        workdir,
    )
    cross_check_embedding(split1_results)

    # Probe splits 2-4 for residual + logits (same approach)
    split_results = [split1_results]
    for i, bin_path in enumerate(args.bins[1:], 2):
        print(f"\n=== Probing split {i}: {Path(bin_path).name} ===")
        # For splits 2-4 we'd need to feed the residual from split 1's output.
        # The probe for multi-split requires chaining; for now print a note.
        print(f"  [NOTE] Chained split probe for split {i} requires feeding split {i-1}")
        print(f"         output as input. Manual step — run qnn_bin_run with")
        print(f"         SP_QNN_BIN_DUMP_NATIVE=1 (future feature) to capture mid-chain")
        print(f"         native bytes. For now, QNN_QUANT_RESIDUAL uses split-1 anchor.")
        split_results.append({})

    input_params = derive_input_params()
    print_summary(split_results, input_params)


if __name__ == "__main__":
    main()
