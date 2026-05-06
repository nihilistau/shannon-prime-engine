#!/system/bin/sh
# device_dma_init.sh — run on device before DMA probe / Phase 9 tests
# Usage: adb shell sh /data/local/tmp/sp-engine/device_dma_init.sh
#
# Sets up DMA heap / ION / FastRPC permissions for non-root inference testing.
# Run this once per boot, before any sp-engine test that needs DMA alloc.
#
# Requirements:
#   - adb root (or the process must already have root)
#   - Android 12+ for /dev/dma_heap/system (ION path used on older)
#
# What each step does:
#   chmod 666 /dev/dma_heap/system  — allow user-space DMA heap alloc
#   chmod 666 /dev/ion              — fallback for Android <= 11
#   setenforce 0                    — SELinux permissive (research only)
#   setprop grp.adsprpc.unauth_enable 1  — allow unauthenticated FastRPC

echo "[dma_init] Starting DMA permission setup..."

# ── DMA heap ──────────────────────────────────────────────────────────────
if [ -e /dev/dma_heap/system ]; then
    chmod 666 /dev/dma_heap/system && \
        echo "[dma_init] /dev/dma_heap/system: 666 OK" || \
        echo "[dma_init] WARNING: chmod /dev/dma_heap/system failed (need root)"
else
    echo "[dma_init] /dev/dma_heap/system not found (Android < 12?)"
fi

# ── ION fallback ──────────────────────────────────────────────────────────
if [ -e /dev/ion ]; then
    chmod 666 /dev/ion && \
        echo "[dma_init] /dev/ion: 666 OK" || \
        echo "[dma_init] WARNING: chmod /dev/ion failed"
else
    echo "[dma_init] /dev/ion not present (Android 12+ normal)"
fi

# ── FastRPC unauthenticated access ────────────────────────────────────────
setprop grp.adsprpc.unauth_enable 1
echo "[dma_init] setprop grp.adsprpc.unauth_enable 1"

# ── SELinux permissive (research only — removes policy enforcement) ───────
# WARNING: This disables ALL SELinux enforcement. Only use for research.
# Comment out for production or when testing the policy path.
if [ "${SP_SELINUX_PERMISSIVE:-0}" = "1" ]; then
    setenforce 0 && \
        echo "[dma_init] SELinux: PERMISSIVE (research mode)" || \
        echo "[dma_init] WARNING: setenforce 0 failed"
else
    echo "[dma_init] SELinux: enforcing (set SP_SELINUX_PERMISSIVE=1 to disable)"
fi

# ── ADSP library path ─────────────────────────────────────────────────────
ADSP_PATH="/data/local/tmp/sp-engine;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"
echo "[dma_init] ADSP_LIBRARY_PATH will be: ${ADSP_PATH}"

# ── Verify /sys/kernel/debug/dma_buf/bufinfo ─────────────────────────────
if [ -e /sys/kernel/debug/dma_buf/bufinfo ]; then
    echo "[dma_init] bufinfo accessible — listing current DMA-BUF allocations:"
    cat /sys/kernel/debug/dma_buf/bufinfo 2>/dev/null | head -20
else
    echo "[dma_init] bufinfo not accessible (need debugfs mounted or root)"
fi

echo "[dma_init] Done. You can now run sp-engine with DMA alloc."
echo ""
echo "  LD_LIBRARY_PATH=/data/local/tmp/sp-engine:/data/local/tmp/sp22u/qnn \\"
echo "  ADSP_LIBRARY_PATH='${ADSP_PATH}' \\"
echo "  /data/local/tmp/sp-engine/sp-engine <verb> ..."
