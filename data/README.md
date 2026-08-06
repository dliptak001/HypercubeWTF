# MNIST data (HypercubeWTF)

`wtf_mnist` loads **only** from:

```text
C:\HypercubeWTF\data
```

That is the local deploy root (same place the optional CMake POST_BUILD copies
`wtf_mnist.exe`). It does **not** use this CLion/source-tree `data/` folder,
cwd walks, or other projects (e.g. HypercubeCNN).

## Required files (uncompressed IDX)

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

These are **not** committed. Populate `C:\HypercubeWTF\data` once.

### Download (example)

From `C:\HypercubeWTF\data`:

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

(On Windows you can use any tool that fetches and gunzips those four files into that folder.)
