// Phase 1 smoke: construct WTF with tiny legal dim and exit 0.
#include "WTF.h"

#include <cstdio>
#include <cstdlib>

int main()
{
    WTFConfig cfg;
    cfg.reservoir.dim = 5; // N = 32
    cfg.reservoir.history_depth = 4;
    cfg.reservoir.seed = 1;
    cfg.reservoir.verbose = false;
    cfg.ic_seed = 2;
    cfg.episode.T = 0;              // → N
    cfg.episode.readout_slices = 1; // B = 1
    cfg.readout.dim = 0;            // auto
    cfg.readout.num_outputs = 2;
    cfg.readout.task = ReadoutTask::Classification;

    try
    {
        WTF wtf(cfg);
        if (wtf.N() != 32 || wtf.T() != 32 || wtf.B() != 1 || wtf.M() != 4)
        {
            std::fprintf(stderr, "wtf_smoke: unexpected sizes N=%zu T=%zu B=%zu M=%zu\n",
                         wtf.N(), wtf.T(), wtf.B(), wtf.M());
            return 1;
        }
        std::printf("wtf_smoke: ok N=%zu T=%zu B=%zu M=%zu\n",
                    wtf.N(), wtf.T(), wtf.B(), wtf.M());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "wtf_smoke: %s\n", e.what());
        return 1;
    }
}
