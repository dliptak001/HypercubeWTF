#include "WTF.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace {

uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace

WTF::WTF(const WTFConfig& cfg)
    : ic_seed_(cfg.ic_seed),
      readout_cfg_(cfg.readout)
{
    reservoir_ = Reservoir::Create(cfg.reservoir);
    n_ = reservoir_->Size();
    M_ = reservoir_->HistoryDepth();

    T_ = cfg.episode.T == 0 ? n_ : cfg.episode.T;
    if (T_ == 0)
        throw std::invalid_argument("WTF: episode T must be > 0");

    B_ = cfg.episode.readout_slices;
    if (B_ == 0 || (B_ & (B_ - 1)) != 0)
        throw std::invalid_argument("WTF: readout_slices (B) must be a power of two >= 1");
    if (B_ > M_)
        throw std::invalid_argument("WTF: readout_slices (B) must be <= history_depth (M)");

    // Readout dim must match multi-slice pack: features = B * N = 2^readout.dim
    // when B is power of two and N = 2^res.dim → readout.dim = res.dim + log2(B).
    size_t log2_B = 0;
    for (size_t b = B_; b > 1; b >>= 1)
        ++log2_B;
    const size_t expected_readout_dim = reservoir_->Dim() + log2_B;
    if (readout_cfg_.dim == 0)
        readout_cfg_.dim = expected_readout_dim;
    else if (readout_cfg_.dim != expected_readout_dim)
        throw std::invalid_argument(
            "WTF: readout.dim must be 0 (auto) or reservoir.dim + log2(B)");

    // Frozen s0 ~ U[-0.5, 0.5] on N×M, from ic_seed only.
    s0_.assign(n_ * M_, 0.0f);
    std::mt19937_64 rng(mix64(ic_seed_ ^ 0x5343000000000001ULL));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& v : s0_)
        v = dist(rng);
}
