# ML Three Particle

Utilities and notebooks for reconstructing Lambda decay vertex information from three-particle ZDC detector inputs. The current workflow builds WSi image stacks for two photons and one neutron, then trains a PyTorch CNN regressor on the generated calorimeter images.

## Contents

- `test_three_particle_ml.py`: dataset, image-building, geometry, and CNN training utilities.
- `zdc_mamba_baseline.py`: supervised sparse-token ZDC Mamba baseline for momentum regression.
- `CNN-notebook.ipynb`: exploratory CNN workflow.
- `CNN-notebook-wsi-sipm-comparison-v3.ipynb`: WSi/SiPM comparison notebook.

## Data Inputs

The Python script expects ROOT files with an `events` tree and `MCParticles` branches. By default, the configured paths are:

- `../photon1_test_50000.root`
- `../photon2_test_50000.root`
- `../neutron_test_50000.root`

These data files are intentionally ignored by Git because they are typically large and environment-specific.
Generated reports, including PDFs, are also ignored.

## Environment

Create a Python environment and install the analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy torch uproot awkward matplotlib jupyter
```

Use a CUDA-enabled PyTorch install if training should run on GPU. For the Mamba
baseline, install `mamba-ssm` and its CUDA dependencies. The module has a GRU
fallback only so CPU smoke tests can import and run; production physics
comparisons should use the real Mamba path on CUDA.

## Usage

Open the notebooks for exploratory analysis:

```bash
jupyter notebook
```

Or import the training utilities from `test_three_particle_ml.py` in a script or notebook:

```python
from test_three_particle_ml import train_model

model, z_mean, z_std = train_model(train_events, val_events)
```

The expected event dictionaries contain `gamma1`, `gamma2`, and `neutron` hit arrays plus a `z_vertex` target.

## Sparse Mamba Baseline

The first sparse supervised baseline lives in `zdc_mamba_baseline.py` and is
kept separate from the working CNN code. It consumes the same per-event hit
dictionaries for `gamma1`, `gamma2`, and `neutron`, but expects a momentum
target either as top-level `momentum = [px, py, pz]` or top-level `px`, `py`,
`pz`.

```python
from zdc_mamba_baseline import (
    ZDCMambaConfig,
    train_zdc_mamba_baseline,
    validate_serialization_examples,
)

cfg = ZDCMambaConfig()
cfg.tokenization.method = "layer_hilbert"  # also: "layer_raster", "shuffled"
cfg.target.mode = "magnitude_direction"    # also: "cartesian"

reports = validate_serialization_examples(train_events, cfg.tokenization)
model, metadata = train_zdc_mamba_baseline(train_events, val_events, cfg)
```

Serialization convention:

- deposits are aggregated by detector cell `(layer, ix, iy)`;
- only occupied cells are emitted as tokens;
- `layer_hilbert` sorts by increasing detector layer, then by 2D Hilbert index inside that layer;
- the transverse plane is represented as a 64x64 Hilbert grid, matching the existing dense image dimensions and embedding the intended 60x60 active plane;
- `layer_raster` keeps the same features but replaces Hilbert order with ordinary raster order;
- `shuffled` keeps the same features and coordinates but randomizes token order;
- optional one-token-per-populated-layer summaries are configurable and default off.

Token features are `log_energy`, normalized `x`, normalized `y`, normalized
`z`, normalized layer index, normalized within-layer Hilbert index, particle
type, first-token-in-layer flag, Hilbert-index gap from the previous token,
layer-index change from the previous token, and a layer-summary flag.

Model and target:

- the three particle sequences are encoded independently with a shared Mamba-style encoder;
- photon 1 and photon 2 share the photon particle-type embedding, while the neutron uses a separate type embedding;
- pooling concatenates masked mean, masked max, and energy-weighted mean for each particle;
- the preferred head predicts momentum magnitude plus a raw 3-vector direction, normalized with `torch.nn.functional.normalize`;
- the reconstructed Cartesian prediction is `predicted_magnitude * unit_direction`;
- the control head predicts direct Cartesian `(px, py, pz)`.

Losses:

- magnitude loss: Smooth L1 on the configured magnitude representation (`log`, `identity`, or training-set `standardize`);
- direction loss: `1 - dot(unit_pred, unit_true)`;
- total factorized loss: `lambda_p * magnitude_loss + lambda_dir * direction_loss`;
- Cartesian control loss: MSE on `(px, py, pz)`.

Diagnostics include sequence-length summaries, active-token fraction relative
to the dense detector volume, parameter count, epoch timing, separate
validation magnitude/direction losses, Cartesian residuals, momentum-magnitude
residuals, angular separation, and projected direction residuals in `px/pz`
and `py/pz`.

FM4NPP/CLAS12 reuse:

- inspected the local `main` ref of `/home/alessio/ML-work/CLAS12`, especially `fm4npp/hilbert.py`, for the FM4NPP Hilbert encode/decode implementation;
- inspected `fm4npp/models/mambagpt.py` and `train/downstream/model.py` on that ref for Mamba block, residual, normalization, projection, mask, and pooling patterns;
- adapted the same serialization principle used by the current CLAS12 layer-band work to ZDC: explicit layer identity dominates order, while Hilbert is only a within-layer transverse ordering;
- copied the RMSNorm pattern locally to avoid adding a runtime dependency on the separate CLAS12 checkout;
- preserved the ZDC CNN preprocessing conventions for hit arrays, detector layer indices, transverse binning, log energy, GPU temperature guard, and train/validation split ownership.
