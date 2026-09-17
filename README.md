# ZDC photon momentum regression

This repository is a reproducible, script-only sparse-token study for photon
momentum reconstruction from the ZDC WSi and SiPM detectors.  The supported
workflow is:

```bash
zdc audit --config configs/photon.yaml
zdc prepare --config configs/photon.yaml
zdc train --config configs/photon.yaml --run-dir runs/photon_seed1
zdc evaluate --config configs/photon.yaml --run-dir runs/photon_seed1
```

The conceptual hit token is `(E, x, y, z, detector_id)`.  The mmap Ninja
format stores `(E, x, y, z)` as a ragged float array and the detector ID as a
compact parallel sidecar, recreating the five-field token only while loading.
Training fits all normalization values from the training split only: a separate
min--max range for each detector and a combined global coordinate range.  This
gives the model detector-local coordinates in `[-1, 1]` without discarding the
relative placement of WSi and SiPM.

The input projection concatenates log-scaled energy, local coordinates, global
coordinates, two Fourier/NeRF-style sin/cos bands of the local coordinates,
and an 8-dimensional learned detector embedding.  The loader first merges
exact duplicate cells, then serializes by detector, depth layer (20 WSi or 64
SiPM bins), and 64x64 Hilbert order within each layer.  ROOT collection order
therefore cannot define the Mamba sequence.

`model.max_tokens` is an optional event-local compression cap, not a fixed
padding length.  Set it to `null` to retain every post-merge hit; batches still
right-pad only to their own longest event.  `model.input_normalization` controls
only the raw feature `LayerNorm(27)` before the first projection (`layernorm` or
`none`); it does not disable the later latent `LayerNorm(dim)` before pooling.

`model.event_summary: calorimeter` is a matched accessibility ablation.  It
leaves the token encoder and attention pooling unchanged, but gives the final
head four additional whole-event quantities computed before any token cap:
`log1p(sum E_WSi / median_train_sum_E_WSi)`, the analogous SiPM sum, and the
two `log1p` merged-cell counts.  This is not a replacement for a calibrated
energy baseline: the same joint log-Huber loss and three-output head remain.
It tests whether explicit calorimetric aggregates improve the learned energy
response under otherwise identical training.

The matched medium ablation configurations are `photon_medium.yaml` (baseline),
`photon_medium_uncapped.yaml`, `photon_medium_no_raw_ln.yaml`, and
`photon_medium_uncapped_no_raw_ln.yaml`.
`photon_medium_calorimeter_summary.yaml` is the corresponding baseline-matched
calorimeter-summary run.

The energy target normally uses `training.energy_objective: log`, namely
Huber or MAE on standardized `log(E)` together with the two standardized angle
terms.  The matched `relative_energy_huber` ablation instead uses
`Huber(((E_pred - E_true) / E_true) / 0.10)` for energy and leaves the angular
Huber terms unchanged.  The fixed `0.10` is a 10% fractional-residual scale:
it sets both the Huber transition and a dimensionless relative-energy unit.
The MAE ablation changes the existing standardized three-target objective from
Huber to absolute error, without changing the target parameterization.
`photon_medium_relative_energy_huber_calorimeter_summary.yaml` combines the
relative-energy Huber objective with the calorimeter-summary head input; it is
the direct matched test of whether those exact sums add information beyond the
sequence encoder under the response-aligned objective.

## Clean uncapped photon study

The five controlled arms in `configs/photon_clean_*.yaml` use uncapped tokens,
no raw-input LayerNorm, the same deterministic medium split, and a 96-D,
four-layer Mamba trunk.  The neural joint arms use distinct energy and angular
heads.  In the summary arm, calorimeter totals/counts enter the energy head
only.  This separates information-access and multitask questions from the
earlier exploratory configurations.

```bash
zdc sum-baseline --config configs/photon_clean_sum_baseline.yaml --run-dir runs/photon_clean_sum_baseline
zdc train --config configs/photon_clean_energy_only.yaml --run-dir runs/photon_clean_energy_only
zdc train --config configs/photon_clean_angles_only.yaml --run-dir runs/photon_clean_angles_only
zdc train --config configs/photon_clean_joint.yaml --run-dir runs/photon_clean_joint
zdc train --config configs/photon_clean_joint_calorimeter_summary.yaml --run-dir runs/photon_clean_joint_calorimeter_summary
```

Evaluate each neural arm with its matching `zdc evaluate` invocation.  The
two-sum command already writes `evaluation_test.json` and test predictions.

Splits are deterministic by original source-event index: 70% train, 15%
validation, and 15% held-out test by default.  Matching event indices from the
two photon source files always receive the same split.  Training uses only the
first two partitions and saves the validation-best checkpoint; `zdc evaluate`
uses split 2 (test) unless explicitly changed in Python.

Run local audits and CPU smoke tests with
`/home/alessio/miniconda3/envs/fm4npp/bin/python`.  Production training runs
inside the edgexpert Docker environment; copy the rendered configuration and
prepared dataset there, then collect artifacts through the mounted
`/home/alessio/mnt/edgexpert-a02c/` view.

`legacy/` contains the retired notebooks and prototype comparison.  It is not
part of the supported workflow or performance baseline.

## Environment handoff

On edgexpert, enter the established training Docker environment first, then
run its preflight before starting a campaign:

```bash
bash scripts/edgexpert_preflight.sh
zdc audit --config configs/photon.yaml
zdc prepare --config configs/photon.yaml
zdc campaign --config configs/photon.yaml --output-dir campaigns/photon_v1 --seeds 1 2 3
```

The campaign command renders one configuration per seed. Launch each pending
run with `zdc train --config campaigns/photon_v1/seed_0001/config.yaml
--run-dir campaigns/photon_v1/seed_0001`, then evaluate it with the analogous
`zdc evaluate` command. Do not launch a training run until the audit reports
an exactly-one-photon truth contract for both source files.
