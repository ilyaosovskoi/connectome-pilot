# Roadmap — from today's prototype to a multi-body self-improving pilot

Target: a small connectome brain that (1) drives different robots,
(2) anyone can fine-tune, (3) improves itself while working.
Status: Phase 0 done (pipeline, honest metrics, turn path woken).

## Phase 1 — Steering that works (prerequisite for everything)

- [ ] 1.1 Bigger validation (16+ episodes) so selection stops chasing noise.
- [ ] 1.2 Full-brain gain selection (background, hours) — or lock 256 as
  the workhorse with the full brain as signal-quality reference.
- [ ] 1.3 Transmission objectives: reward DN-asymmetry correlation with the
  teacher turn, not just behavior (behavior reward can't reach a quiet path).
- [ ] 1.4 Targeted recruitment: raise sensor quota for loom/turn afferents
  at extraction; verify DNa afferents first.
- [ ] **Gate:** DN-readout honest task ≥ 50% (expert 88%). Below this,
  phases 2-4 build on sand.

## Phase 2 — One brain, many bodies

- [ ] 2.1 Freeze the body interface (4 sensors / 2 motors) as v1 API.
- [ ] 2.2 Body zoo: rover, arm (done) + corridor-follower, beacon-homing
  vacuum sim, 1-2 more from `docs/useful_robots.md`.
- [ ] 2.3 Shared plastic core + per-body readout heads (checkpoints per body).
- [ ] 2.4 Transfer metric: zero-shot progress + few-demo success per new body.
- [ ] **Gate:** a new body reaches 50%+ with < 1000 demo ticks.

## Phase 3 — Fine-tuning for everyone (CLI first)

- [ ] 3.1 Single entry point: `flypilot fit --body rover --demos ...`,
  `flypilot eval`, `flypilot run --checkpoint`, `flypilot export --mcu`.
  (Thin wrappers over robot/*.py + YAML configs instead of 20 flags.)
- [ ] 3.2 `flypilot demo`: 5-minute quickstart with frozen numbers.
- [ ] 3.3 Docs + contribution guide + data-license checklist per release.
- [ ] 3.4 (later) Minimal web UI: upload demo video/telemetry -> checkpoint.
- [ ] **Gate:** a stranger files an issue saying they fine-tuned a body.

## Phase 4 — Self-improvement during operation

- [ ] 4.1 Port plasticity + homeostasis + novelty to `firmware/flynet.h`
  (today Python-only) — the brain must learn without the laptop.
- [ ] 4.2 Safety cage: spinal-reflex overrides (bumper/cliff), weight bounds,
  keep-best rollback to last validated checkpoint on performance drop.
  Rule: never accept a change that wasn't validated — in production too.
- [ ] 4.3 `selftune.py`: hyperparameter search with keep-best (learn to learn).
- [ ] 4.4 (later) Fleet learning: share validated checkpoints across robots.
- [ ] **Gate:** a robot deployed for a week is better than on day one,
  with logs to prove it.

## Standing rules (from experience)

- Closed-loop selection only; clone MSE lies.
- Every learning step needs keep-best validation — offline and on-device.
- Report honest numbers, including where hand-code wins.
