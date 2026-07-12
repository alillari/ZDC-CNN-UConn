# ML Three Particle

Utilities and notebooks for reconstructing Lambda decay vertex information from three-particle ZDC detector inputs. The current workflow builds WSi image stacks for two photons and one neutron, then trains a PyTorch CNN regressor on the generated calorimeter images.

## Contents

- `test_three_particle_ml.py`: dataset, image-building, geometry, and CNN training utilities.
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

Use a CUDA-enabled PyTorch install if training should run on GPU.

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
