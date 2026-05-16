// Shannon-Prime Engine — FFN (Phase 1.6 skeleton).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// SwiGLU FFN in O_K coordinates: out = (silu(gate_proj(x)) * up_proj(x)) @ down_proj.

#pragma once

#include "sp_ok_tensor.h"

namespace sp::engine {

// SwiGLU FFN: out = down_proj(silu(gate_proj(x)) * up_proj(x))
//
// SKELETON: signature only. Phase 1.6 fills in the integer-coordinate
// matmul + silu approximation.
void sp_ffn_swiglu(const sp_ok_tensor& x,
                    const sp_ok_tensor& gate_w,
                    const sp_ok_tensor& up_w,
                    const sp_ok_tensor& down_w,
                    sp_ok_tensor&       out);

}  // namespace sp::engine
