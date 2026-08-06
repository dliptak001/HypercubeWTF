#pragma once

// MNIST data location for demos. Fixed local deploy root only:
//   C:\HypercubeWTF\data
// (same tree as the optional CMake POST_BUILD copy of wtf_mnist.exe).
// Never walks the CLion / source-tree clone. Example-only helper.

#include <filesystem>
#include <stdexcept>
#include <string>

namespace wtf_ex {

inline bool DirHasMnistIdx(const std::filesystem::path& dir)
{
    namespace fs = std::filesystem;
    return fs::is_directory(dir)
           && fs::exists(dir / "train-images-idx3-ubyte")
           && fs::exists(dir / "train-labels-idx1-ubyte")
           && fs::exists(dir / "t10k-images-idx3-ubyte")
           && fs::exists(dir / "t10k-labels-idx1-ubyte");
}

/// Resolve C:\HypercubeWTF\data. @p argv0 unused (kept for call-site stability).
inline std::filesystem::path FindMnistDataDir(const char* /*argv0*/)
{
    namespace fs = std::filesystem;
    const fs::path data = fs::path("C:/HypercubeWTF/data");

    if (DirHasMnistIdx(data))
        return fs::weakly_canonical(data);

    throw std::runtime_error(
        "Cannot find MNIST IDX files in C:\\HypercubeWTF\\data\n"
        "Need:\n"
        "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
        "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n"
        "See examples/README.md (Appendix: MNIST files)");
}

} // namespace wtf_ex
