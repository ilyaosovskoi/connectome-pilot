# Data license

Code in this repository is MIT (see `LICENSE`). **Data files derived from
FlyWire are NOT MIT.**

- `data/fly_circuit_*.json`, `data/fly_circuit_*.npz`, `data/sensorimotor.npz`,
  `data/neuron_atlas.json`, `data/mushroom_body_neurons.json`,
  `data/2025_Completeness_783.csv` are derived from the FlyWire connectome
  (FAFB v783) and are shared under **CC BY-NC 4.0** (non-commercial use,
  attribution required).
- The full connectome (`2025_Connectivity_783.parquet`,
  `flywire_annotations.tsv`, fetched by `scripts/download_full.py`) is
  governed by the FlyWire terms of use (https://flywire.ai) and is **not**
  committed to this repo (see `.gitignore`).

If you use the derived data, please cite:

- Dorkenwald, S. et al. *Neuronal wiring diagram of an adult brain.*
  Nature (2024).
- Shiu, P. et al. *A leaky integrate-and-fire computational model of the
  whole fly brain.* Nature (2024).
- FlyWire Codex: https://codex.flywire.ai
