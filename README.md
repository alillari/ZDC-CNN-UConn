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
