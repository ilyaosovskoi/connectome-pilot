# Domovoy FR-1 — home rover for the connectome pilot

A small home robot whose controller is the fly-connectome pilot
(`robot/run_robot.py --bodies rover`). Not a concept: every line below maps
to an implemented channel, a measured number, or an exported file.

## Mission

Night patrol of an apartment: visit rooms in turn (beacon in each room),
don't hit furniture, pets, or stairs. Learns the specific apartment instead
of needing a map — the same binary that works in a studio adapts to a
three-room flat through on-chip plasticity.

## Body

- Differential 2WD, 200 mm diameter, 1.2 kg, max 0.3 m/s (safe around pets).
- 2× front IR beacon photodiodes (left/right) -> `odor_l/r`: bearing toward
  the active room beacon. The gradient asymmetry is the steering signal
  (measured: correlates -0.83 with expert turn commands).
- 2× ToF rangefinders (left/right, 3.5 m) -> `loom_l/r`: furniture and pets.
- Bumper ring + 3× cliff sensors -> terminal crash signal (PPL1 channel).
- Dock IR receiver -> success signal (PAM channel): reached the room.

## Brain on board

- STM32G4 (170 MHz, 128 KB flash, 32 KB RAM). No Linux, no radio needed.
- `firmware/flynet.h`, circuit-256: ~20 KB flash, ~6.4 KB RAM (measured).
- Loop: read 4 sensors -> 10 LIF steps x 5 ms -> 2 wheel commands, 20 Hz.
- On-chip learning: R-STDP on 1015 excitatory synapses (bounded [0.2x, 5x])
  + homeostasis + novelty bonus. Factory image holds the teacher-trained
  checkpoint (`23% -> 27%` sim success, `8%` crash); the apartment finishes
  the training.
- Safety is NOT in the brain: independent hard cutoff on bumper/cliff that
  the network cannot override (like a spinal reflex).

## Why this body fits this brain

- The task is the sim task: beacon-seeking with obstacles. Sim-to-real gap
  is one calibration (gains odor x loom, selected closed-loop).
- Novelty bonus becomes room coverage: unvisited corners pay, pacing pays
  nothing — measured 36 states then decay in sim.
- PAM/PPL split matches hardware: dock sensor vs bumper, no reward hacking.

## Roadmap to siblings

Same board + same header, bigger chassis: warehouse AMR (beacon = pick
station), farm row rover (beacon = row-end reflector, loom = crop stems).
The arm (`robot/embodiments.py Arm`, 30% sim reach) becomes the tabletop
sibling when its turn pathway is repaired.
