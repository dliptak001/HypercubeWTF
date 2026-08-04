#pragma once

// Example-only packing helpers (caller-owned field → length-N drive).
// Not part of the WTF core library.
//
// Locked demo packers (v1):
//   1) Pad / low addresses — write data into verts [0, P), pad [P, N).
//   2) DualPlane (MNIST) — ink || |grad| via HCNN SpatialEmbed DualPlaneResize;
//      occupied verts are low addresses; any tail is pad_value.
//
// Explicitly out of scope: standalone resize-to-N, letterbox/center-crop as a
// product path, bit-address (row,col)→vertex maps.

#include "HCNNSpatialEmbed.h"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace wtf_ex {

/// Fill @p out (length N) with @p data in low addresses; pad the rest.
/// Requires data.size() <= out.size().
inline void PackPadLow(std::span<const float> data, std::span<float> out,
                       float pad_value = -1.0f)
{
    if (data.size() > out.size())
        throw std::invalid_argument(
            "PackPadLow: data longer than field N (need pad room or larger dim)");
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = data[i];
    for (size_t i = data.size(); i < out.size(); ++i)
        out[i] = pad_value;
}

/// MNIST DualPlane pack: 28×28 in [-1,1] → length N = 2^dim (low addresses).
/// Uses vendored HCNN SpatialEmbed DualPlaneResize (ink || |grad| + pad tail).
inline void PackDualPlane28(const float* img28x28, int dim, std::span<float> out,
                            float pad_value = -1.0f)
{
    if (img28x28 == nullptr)
        throw std::invalid_argument("PackDualPlane28: null image");
    if (dim < 5 || dim > 16)
        throw std::invalid_argument("PackDualPlane28: dim must be in [5, 16]");

    hcnn::HCNNSpatialEmbedConfig ec;
    ec.dim = dim;
    ec.mode = hcnn::HCNNSpatialEmbedMode::DualPlaneResize;
    ec.pad_value = pad_value;
    ec.plane_side = 0; // auto S from N
    hcnn::HCNNSpatialEmbedder emb(ec);

    const int N = emb.capacity();
    if (static_cast<size_t>(N) != out.size())
        throw std::invalid_argument(
            "PackDualPlane28: out.size() must equal N = 2^dim");

    emb.embed(img28x28, 28, 28, out.data());
}

/// Build a DualPlane embedder (for logging plan / capacity).
inline hcnn::HCNNSpatialEmbedder MakeDualPlaneEmbedder(int dim,
                                                       float pad_value = -1.0f)
{
    hcnn::HCNNSpatialEmbedConfig ec;
    ec.dim = dim;
    ec.mode = hcnn::HCNNSpatialEmbedMode::DualPlaneResize;
    ec.pad_value = pad_value;
    ec.plane_side = 0;
    return hcnn::HCNNSpatialEmbedder(ec);
}

} // namespace wtf_ex
