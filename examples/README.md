# HypercubeWTF examples

This folder holds **two runnable demos** of HypercubeWTF.

**What the product does (in plain terms):** take a fixed-size array of numbers
(a “field”), run it through a **frozen** recurrent network on a hypercube
(the *reservoir*), then train a small **HypercubeCNN** head to classify the
result. The reservoir weights never learn; only the head does.

These demos own everything outside that core: inventing synthetic inputs,
loading MNIST, laying images onto the field, and optional noise experiments.
The library itself is collect → train → predict.

Build the project in **Release** (CLion or `cmake --build cmake-build-release`),
then run the binary you care about.

| Program | What it is for | Do you need data files? |
|---------|----------------|-------------------------|
| `wtf_synth` | Quick check that the stack works | No |
| `wtf_mnist` | Real images (handwritten digits) and the written studies | Yes — see below |

Settings live at the **top of each program’s `.cpp` file**: product options in
`MakeWTFConfig()`, and a few demo-only constants just under that (how many
samples, optional noise switches, pass/fail floor for continuous integration).

---

## `wtf_synth` — run this first

A self-contained smoke test. It **builds fake multi-class patterns in memory**
(no download), trains the head, and scores new patterns it has not trained on.

You should see a test accuracy well above the soft floor of **0.70** (often
near ceiling under the shipped settings). If it fails that floor, something is
wrong with the build or runtime.

Source: [`synth/wtf_synth.cpp`](synth/wtf_synth.cpp).

---

## `wtf_mnist` — handwritten digits

Same train-and-score loop as synth, but the inputs are **MNIST digit images**.
Each 28×28 image is laid onto the hypercube field, the reservoir (or a
pack-only shortcut) produces features, and the head predicts digit 0–9.

### Data setup

`wtf_mnist` does **not** read MNIST from this git clone. It loads **only** from
the local deploy folder:

```text
C:\HypercubeWTF\data\
```

Create that folder if needed, download the standard MNIST IDX files into it
(see [Appendix: MNIST files](#appendix-mnist-files)), and leave the four
uncompressed `*-ubyte` files there. The dataset is **not** part of this
repository.

### About the accuracy numbers

The MNIST demo and study write-ups use a **small** hypercube (dimension 10 →
1024 vertices) so experiments stay quick. Do **not** treat those scores as the
best this family of models can do. HypercubeCNN alone has already shown about
**99.5%** on MNIST; this example is not trying to match that leaderboard.

Source: [`mnist/wtf_mnist.cpp`](mnist/wtf_mnist.cpp).

### Optional studies (longer reads)

Two markdown write-ups use `wtf_mnist` as a vehicle. They answer product
questions; they are not required to run the demo.

| Document | In plain English |
|----------|------------------|
| [`mnist/WhiteNoiseFilter.md`](mnist/WhiteNoiseFilter.md) | If we add strong white noise to the **test** field after packing, does running the reservoir help more than feeding the noisy pack straight to the CNN head? |
| [`mnist/TrainingDataQualitySensitivity.md`](mnist/TrainingDataQualitySensitivity.md) | If the **training** images are deliberately degraded, how much does each path lose on clean or noisy tests? |

Both compare two feature paths that share the same packing and head:

- **Bypass** — skip the reservoir; the packed field goes to the head.
- **Reservoir** — run the short frozen episode; the end state goes to the head.

---

## Folder layout

```text
examples/
  README.md                 # you are here
  common/                   # shared helpers for the demos
  synth/wtf_synth.cpp       # synthetic smoke demo
  mnist/                    # MNIST demo + study docs
```

To add another demo: create `examples/<name>/`, add a target in the root
`CMakeLists.txt`, and reuse `common/` if it helps.

---

## Appendix: MNIST files

**Location (required by the demo):** `C:\HypercubeWTF\data\`

**Required files** (uncompressed IDX, exact names):

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

These are the usual public MNIST binaries (LeCun et al.). We do **not** ship
them in git — fetch once onto your machine.

**Download example** (run from `C:\HypercubeWTF\data`, or save into that folder):

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

On Windows, any tool that downloads those four `.gz` files and decompresses them
into `C:\HypercubeWTF\data` is fine.
