// Shannon-Prime Engine — Gated DeltaNet recurrent state cache (implementation)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).

#include "gdn_state.h"

#include <cstdio>
#include <cstring>

namespace sp::engine {

struct GdnStateCache::Impl {
    // Shape (same for every GDN layer in a given model).
    int conv_kernel   = 0;
    int conv_channels = 0;
    int head_v_dim    = 0;
    int num_v_heads   = 0;
    int n_seqs        = 0;

    // Per-layer backing storage. For non-GDN layers both vectors stay
    // empty; is_gdn[i] records the classification for summary output.
    std::vector<std::vector<float>> conv_state;   // size n_layer
    std::vector<std::vector<float>> ssm_state;    // size n_layer
    std::vector<bool>               is_gdn;       // size n_layer
    int n_gdn = 0;

    int conv_floats_per_seq() const { return (conv_kernel - 1) * conv_channels; }
    int ssm_floats_per_seq()  const { return head_v_dim * head_v_dim * num_v_heads; }
    int conv_floats_total()   const { return conv_floats_per_seq() * n_seqs; }
    int ssm_floats_total()    const { return ssm_floats_per_seq()  * n_seqs; }
};

GdnStateCache::GdnStateCache() : impl_(std::make_unique<Impl>()) {}
GdnStateCache::~GdnStateCache() = default;

std::unique_ptr<GdnStateCache> GdnStateCache::create(
        const std::vector<bool>& layer_is_gdn,
        int conv_kernel,
        int conv_channels,
        int head_v_dim,
        int num_v_heads,
        int n_seqs) {
    if (conv_kernel < 2 || conv_channels < 1 ||
        head_v_dim  < 1 || num_v_heads   < 1 || n_seqs < 1) {
        std::fprintf(stderr,
            "[sp-engine] GdnStateCache::create: invalid shape "
            "(conv_kernel=%d conv_channels=%d head_v_dim=%d num_v_heads=%d n_seqs=%d)\n",
            conv_kernel, conv_channels, head_v_dim, num_v_heads, n_seqs);
        return nullptr;
    }

    auto c = std::unique_ptr<GdnStateCache>(new GdnStateCache());
    auto& I = *c->impl_;
    I.conv_kernel   = conv_kernel;
    I.conv_channels = conv_channels;
    I.head_v_dim    = head_v_dim;
    I.num_v_heads   = num_v_heads;
    I.n_seqs        = n_seqs;

    const size_t n_layer = layer_is_gdn.size();
    I.is_gdn = layer_is_gdn;
    I.conv_state.resize(n_layer);
    I.ssm_state.resize(n_layer);

    const int conv_total = I.conv_floats_total();
    const int ssm_total  = I.ssm_floats_total();
    for (size_t il = 0; il < n_layer; ++il) {
        if (!layer_is_gdn[il]) continue;
        I.conv_state[il].assign((size_t)conv_total, 0.0f);
        I.ssm_state [il].assign((size_t)ssm_total,  0.0f);
        ++I.n_gdn;
    }
    return c;
}

// --- accessors ------------------------------------------------------------
int GdnStateCache::n_layer()       const { return (int)impl_->is_gdn.size(); }
int GdnStateCache::conv_kernel()   const { return impl_->conv_kernel; }
int GdnStateCache::conv_channels() const { return impl_->conv_channels; }
int GdnStateCache::head_v_dim()    const { return impl_->head_v_dim; }
int GdnStateCache::num_v_heads()   const { return impl_->num_v_heads; }
int GdnStateCache::n_seqs()        const { return impl_->n_seqs; }
int GdnStateCache::conv_state_floats() const { return impl_->conv_floats_per_seq(); }
int GdnStateCache::ssm_state_floats()  const { return impl_->ssm_floats_per_seq(); }
int GdnStateCache::n_gdn_layers()      const { return impl_->n_gdn; }

bool GdnStateCache::is_gdn_layer(int layer) const {
    if (layer < 0 || layer >= (int)impl_->is_gdn.size()) return false;
    return impl_->is_gdn[(size_t)layer];
}

// --- state I/O ------------------------------------------------------------
bool GdnStateCache::read_conv(int layer, std::vector<float>& buf) const {
    if (layer < 0 || layer >= (int)impl_->is_gdn.size()) return false;
    if (!impl_->is_gdn[(size_t)layer]) { buf.clear(); return true; }
    buf = impl_->conv_state[(size_t)layer];
    return true;
}

bool GdnStateCache::write_conv(int layer, const float* src) {
    if (layer < 0 || layer >= (int)impl_->is_gdn.size()) return false;
    if (!impl_->is_gdn[(size_t)layer]) return true;
    if (!src) return false;
    auto& dst = impl_->conv_state[(size_t)layer];
    std::memcpy(dst.data(), src, dst.size() * sizeof(float));
    return true;
}

bool GdnStateCache::read_ssm(int layer, std::vector<float>& buf) const {
    if (layer < 0 || layer >= (int)impl_->is_gdn.size()) return false;
    if (!impl_->is_gdn[(size_t)layer]) { buf.clear(); return true; }
    buf = impl_->ssm_state[(size_t)layer];
    return true;
}

bool GdnStateCache::write_ssm(int layer, const float* src) {
    if (layer < 0 || layer >= (int)impl_->is_gdn.size()) return false;
    if (!impl_->is_gdn[(size_t)layer]) return true;
    if (!src) return false;
    auto& dst = impl_->ssm_state[(size_t)layer];
    std::memcpy(dst.data(), src, dst.size() * sizeof(float));
    return true;
}

void GdnStateCache::reset() {
    for (auto& v : impl_->conv_state) if (!v.empty()) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : impl_->ssm_state)  if (!v.empty()) std::fill(v.begin(), v.end(), 0.0f);
}

void GdnStateCache::print_summary(std::FILE* f) const {
    const auto& I = *impl_;
    const size_t conv_bytes = (size_t)I.n_gdn * (size_t)I.conv_floats_total() * sizeof(float);
    const size_t ssm_bytes  = (size_t)I.n_gdn * (size_t)I.ssm_floats_total()  * sizeof(float);
    std::fprintf(f, "GdnStateCache:\n");
    std::fprintf(f, "  layers (total / GDN): %d / %d\n", (int)I.is_gdn.size(), I.n_gdn);
    std::fprintf(f, "  conv_kernel:          %d\n", I.conv_kernel);
    std::fprintf(f, "  conv_channels:        %d\n", I.conv_channels);
    std::fprintf(f, "  head_v_dim:           %d\n", I.head_v_dim);
    std::fprintf(f, "  num_v_heads:          %d\n", I.num_v_heads);
    std::fprintf(f, "  n_seqs:               %d\n", I.n_seqs);
    std::fprintf(f, "  per-layer conv state: %d floats (%.1f KiB)\n",
                 I.conv_floats_total(),
                 I.conv_floats_total() * sizeof(float) / 1024.0);
    std::fprintf(f, "  per-layer ssm state:  %d floats (%.1f KiB)\n",
                 I.ssm_floats_total(),
                 I.ssm_floats_total() * sizeof(float) / 1024.0);
    std::fprintf(f, "  total footprint:      %.2f MiB\n",
                 (conv_bytes + ssm_bytes) / (1024.0 * 1024.0));
}

} // namespace sp::engine
