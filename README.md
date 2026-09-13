# Connectome-Pilot — fruit-fly connectome as a small adaptive robot controller

> **Actively developed.** Goal: learn how to use biological experience —
> a real insect connectome — to build **self-improving systems for robots**:
> small controllers that start from evolved wiring and keep adapting
> to new bodies and tasks.

A real piece of *Drosophila* brain (FlyWire v783 connectome: real neuron IDs,
real synapse counts, real transmitter signs) wired as a controller for robots.
The connectome is the **initialization**, not a frozen reservoir: task-driven
plasticity tunes a small synaptic subset, a linear readout maps descending
neurons to motors, and the same brain runs different bodies (rover, arm).

> Status: research prototype. Honest numbers below — including where a
> 3-line hand-coded controller still wins.

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/download_full.py          # full connectome (~134 MB, git-lfs)
python3 scripts/extract_circuit.py --budget 256 --out data/fly_circuit_256.json
python3 robot/formats.py --circuit data/fly_circuit_256.json --emit   # -> .npz
python3 robot/dataset.py --circuit data/fly_circuit_256.npz           # -> sensorimotor.npz
python3 robot/train.py --circuit data/fly_circuit_256.npz --episodes 8
python3 robot/run_robot.py --circuit data/fly_circuit_256.npz --bodies rover,arm --fit-body
```

## Pipeline

| step | script | in → out |
|---|---|---|
| cut a sensorimotor circuit from the full connectome | `scripts/extract_circuit.py` | parquet → `data/fly_circuit_*.json` |
| pick the best runtime format | `robot/formats.py` | json → `*.npz` (CSR, bit-exact) |
| record a sensorimotor dataset | `robot/dataset.py` | world + expert → `data/sensorimotor.npz` |
| train plastic synapses (R-STDP, dopamine-style) | `robot/train.py` | circuit + demos → checkpoint json |
| run the brain as a robot | `robot/run_robot.py` | circuit/checkpoint → rover + arm scores |
| export to microcontroller C header | `robot/export_c.py` | circuit + eval → `firmware/flynet.h` |
| frozen-reservoir baseline + ablations | `robot/eval.py` | real / shuffled / random wiring |

> The browser learning game (Kardashev builder + Skynet eval) used to live
> here; it now lives next door in `../darwin-opus/` (archived visualization).

## 1. Trainable synapses

`robot/plastic.py` — `PlasticFlyCircuit` adds reward-modulated STDP on top of
the frozen connectome:

```
e <- decay * e + outer(post_rate, pre_spikes)   # eligibility trace
W <- clip(W + lr * (r - baseline) * e)          # dopamine-style RPE update
```

Default plastic set: excitatory synapses onto descending (motor-output)
neurons — 62 weights on the 256-neuron circuit. Small enough to stay stable,
large enough to steer behavior (`--scope all-exc` opens ~1000).

`robot/train.py` automates the loop: tune expert → fit readout (frozen
baseline) → R-STDP episodes with held-out validation → re-fit readout →
report delta. Example (circuit-256, 8 episodes):

| | success | crash | fitness |
|---|---|---:|---:|
| frozen baseline | 8% | 33% | 0.101 |
| + R-STDP tuning | 17% | 42% | 0.146 |

Small samples, noisy — the script prints the delta every run so you can see
when plasticity helps and when it just jitters. The honest pattern so far:
plasticity moves behavior (success rate up), crash rate needs a shaping term.

## 2. Best connectome format

`robot/formats.py` benchmarks JSON edge-list vs NPZ-CSR vs int8-quantized
(same synapses/signs, storage only). Circuit-256 (256 neurons, 1304 synapses):

| format | size | load | rollout equivalence |
|---|---|---:|---|
| json edge-list | 62.5 KB | 0.8 ms | — |
| **npz-csr (recommended)** | **6.5 KB** | 0.5 ms | bit-exact (max diff 0.0) |
| npz int8-quant | ~6 KB | 0.5 ms | lossy (small synapses suffer most) |

`FlyCircuit` loads `.json` and `.npz` interchangeably. Datasets and training
use NPZ; JSON stays as the human-inspectable interchange format.

## 3. Dataset

`robot/dataset.py` records expert demonstrations in NPZ form:

```
sensors  (T, 4)  odor_l, odor_r, loom_l, loom_r
actions  (T, 2)  expert left/right motor commands
features (T, F)  frozen-brain firing rates per tick (readout inputs)
reward   (T,)    dense progress reward
episode  (T,)    split (T,)  0=train 1=test (split by episode seed)
```

Default build (circuit-256): 5,833 ticks (train 4,182 / test 1,651),
`data/sensorimotor.npz`, 1.3 MB.

## 4. One brain, two robots

`robot/embodiments.py` + `robot/run_robot.py`. Every body speaks the same
4-sensor / 2-motor language: `Rover` (differential drive, odor + obstacles),
`Arm` (2-link planar reacher; odor = target bearing, loom = joint limits).
Circuit-256, 10 episodes each:

| body | training | success | progress |
|---|---|---:|---:|
| rover | rover demos | 20% | 0.41 |
| arm | zero-shot (rover readout) | 0% | 0.70 |
| arm | **own demos (647 ticks)** | **30%** | 0.69 |

Read the arm row carefully: zero-shot transfer already walks 70% of the way
to the target — the wiring carries a usable sensorimotor prior — but precision
needs body-specific training. A Jacobian expert solves the arm at 100%, so the
task is solvable; the brain is at 30% with 647 demo ticks.

Reference (frozen reservoir, circuit-1024, 24 test episodes): real wiring
reaches 62% vs 58% shuffled / 62% random — no clear win over shuffled on this
reactive task. The connectome's edge is scaling and crash-avoidance at small
sizes (21% crashes vs 38–46% at 128–256 neurons), not raw success rate. A
hand-tuned 3-line Braitenberg controller still wins at 79%. See `robot/eval.py`.

## What is real vs modeled

- **Real:** FlyWire v783 IDs, synapse counts and transmitter signs; LIF
  parameters from Shiu et al. 2024; circuit extraction (flow + fidelity);
  C-port verified numerically against Python.
- **Modeled:** LIF physiology simplified (no NMDA/glia/neuromodulation beyond
  sign); toy sensor physics; linear readout; browser game uses synthetic
  wiring with matched statistics.

## Data

- In repo: `data/neuron_atlas.json`, `data/mushroom_body_neurons.json`,
  circuits `data/fly_circuit_*.json/.npz`, `data/sensorimotor.npz`.
- Download: `python3 scripts/download_full.py` (parquet + annotations,
  git-lfs, resume + sha256). Excluded from git (see `.gitignore`).
- Verify: `python3 scripts/verify_connectome.py` (15,091,983 synapses,
  138,639 neurons; KC→MBON density 11.9%).

## Sources

- Dorkenwald et al., *Nature* 2024 — adult female *Drosophila* connectome
- Shiu et al., *Nature* 2024 — whole-brain LIF simulation
- https://codex.flywire.ai · https://flywire.ai

## License

MIT — see `LICENSE`.
