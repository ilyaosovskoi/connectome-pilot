# Useful robots for a connectome pilot — home, factory, farm

Every body below speaks the same 4-sensor / 2-motor language
(`robot/embodiments.py`): attraction gradient (odor_l/r) + hazard proximity
(loom_l/r) in, differential command (left/right) out. New bodies only need
to translate their sensors and motors into this interface.

## Home

- **Vacuum / mop rover.** Odor = dirt-beacon or room-coverage gradient
  (virtual: distance to nearest uncleaned cell); loom = bumper/lidar.
  Why adaptive: every apartment layout is new; the pilot re-learns
  wall-following vs open-area sweeping instead of shipping a map.
- **Lawn mower.** Odor = boundary-wire / grass-height gradient; loom = trees,
  flowerbeds, pets. Same rover body, different gains — our per-modality
  gains (odor x loom) are exactly this knob.
- **Tabletop arm assistant.** 2-link reacher today; pick-and-place tomorrow.
  Odor = target bearing from wrist camera; loom = joint limits / clutter.

## Production

- **Warehouse AMR.** Odor = RSSI/beacon gradient toward pick station or
  line-following contrast; loom = lidar obstacle field. Why adaptive:
  shelf layouts change nightly; plasticity adapts without re-mapping.
- **Conveyor pick-place arm.** Odor = part-presence heatmap direction;
  loom = human-proximity safety field (slows near people — the same
  hazard channel as obstacle avoidance, different sensor).

## Farm

- **Row-crop weeding rover.** Odor = crop-row signal (NDVI gradient);
  loom = plant-stem proximity (don't crush). Changing crops = new odor
  statistics; R-STDP re-tunes in the field, no cloud needed.
- **Greenhouse harvest arm.** Odor = ripe-fruit color blob bearing;
  loom = trelliswire contact. Delicate, variable targets — the case
  where a pre-wired sensorimotor prior beats training from scratch.

## Why a fly brain (and not a bigger net)

- 256–1024 neurons, 6–100 KB: runs on a Cortex-M4 inside the motor
  controller, no Linux, no radio, no cloud.
- Starts from evolved wiring (not random init): zero-shot rover→arm walk
  already covers 70% of the distance (measured).
- Keeps learning on the job (R-STDP + homeostasis): wear, new floors,
  new crops — same mechanism, no firmware update.
