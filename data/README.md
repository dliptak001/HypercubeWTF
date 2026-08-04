# MNIST data (HypercubeWTF)

`wtf_mnist` loads **only** from this directory (`HypercubeWTF/data/`).  
It does not look at HypercubeCNN or any other project.

## Required files (uncompressed IDX)

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

These are **not** committed (see root `.gitignore`). Populate locally once.

### Download (example)

From this `data/` directory:

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

(On Windows you can use any tool that fetches and gunzips those four files into this folder.)
