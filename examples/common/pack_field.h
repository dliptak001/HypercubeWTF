#pragma once

// Example-only packing helpers (caller-owned field → length-N drive).
// Not part of the WTF core library.
//
// Locked demo packers (v1):
//   1) Pad / low addresses — write data into verts [0, P), pad [P, N).
//   2) DualPlane (MNIST) — ink || |grad| via HCNN SpatialEmbed DualPlaneResize;
//      occupied verts are low addresses; any tail is pad_value.
//   3) PadLowCenter (MNIST, dim=10 / N=1024) — full 28x28 in [0,784) plus a
//      centered 15x16 crop in [784,1024). Simple fixed layout; not general-N.
//
// Explicitly out of scope: standalone resize-to-N product path,
// bit-address (row,col)→vertex maps.

#include "HCNNSpatialEmbed.h"

#include <cstddef>
#include <span>
#include <stdexcept>

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

/// Build a DualPlane embedder (reuse for the whole run — do not construct per sample).
inline hcnn::HCNNSpatialEmbedder MakeDualPlaneEmbedder(int dim,
                                                       float pad_value = -1.0f)
{
    if (dim < 5 || dim > 16)
        throw std::invalid_argument("MakeDualPlaneEmbedder: dim must be in [5, 16]");
    hcnn::HCNNSpatialEmbedConfig ec;
    ec.dim = dim;
    ec.mode = hcnn::HCNNSpatialEmbedMode::DualPlaneResize;
    ec.pad_value = pad_value;
    ec.plane_side = 0; // auto S from N
    return hcnn::HCNNSpatialEmbedder(ec);
}

/// MNIST DualPlane pack using a pre-built embedder (prefer this in hot loops).
inline void PackDualPlane28(const float* img28x28,
                            const hcnn::HCNNSpatialEmbedder& emb,
                            std::span<float> out)
{
    if (img28x28 == nullptr)
        throw std::invalid_argument("PackDualPlane28: null image");
    if (static_cast<size_t>(emb.capacity()) != out.size())
        throw std::invalid_argument(
            "PackDualPlane28: out.size() must equal embedder capacity N");
    emb.embed(img28x28, 28, 28, out.data());
}

// PadLowCenter (MNIST dim=10): full 28x28 + centered 15x16 crop in the tail.
// N must be 1024. Crop origin floor-centered: (row0,col0) = (6,6).
inline constexpr int kPadLowCenterImgSide = 28;
inline constexpr int kPadLowCenterN       = 1024;
inline constexpr int kPadLowCenterCropH   = 15;
inline constexpr int kPadLowCenterCropW   = 16;
inline constexpr int kPadLowCenterRow0 =
    (kPadLowCenterImgSide - kPadLowCenterCropH) / 2; // 6
inline constexpr int kPadLowCenterCol0 =
    (kPadLowCenterImgSide - kPadLowCenterCropW) / 2; // 6

/// Full 28x28 into verts [0,784); centered 15x16 (row-major) into [784,1024).
/// Requires out.size() == 1024. No pad — full occupancy.
inline void PackPadLowCenter28(const float* img28x28, std::span<float> out)
{
    if (img28x28 == nullptr)
        throw std::invalid_argument("PackPadLowCenter28: null image");
    if (out.size() != static_cast<size_t>(kPadLowCenterN))
        throw std::invalid_argument(
            "PackPadLowCenter28: requires N=1024 (dim=10 only for now)");

    constexpr int side = kPadLowCenterImgSide;
    constexpr int n_pix = side * side; // 784
    for (int i = 0; i < n_pix; ++i)
        out[static_cast<size_t>(i)] = img28x28[i];

    constexpr int ch = kPadLowCenterCropH;
    constexpr int cw = kPadLowCenterCropW;
    constexpr int r0 = kPadLowCenterRow0;
    constexpr int c0 = kPadLowCenterCol0;
    float* tail = out.data() + n_pix;
    for (int y = 0; y < ch; ++y)
    {
        const float* row = img28x28 + (r0 + y) * side + c0;
        for (int x = 0; x < cw; ++x)
            tail[y * cw + x] = row[x];
    }
}

} // namespace wtf_ex
