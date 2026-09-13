"""export_c.py — export a circuit + trained readout as a standalone C header.

The result runs on a microcontroller with no Python, no OS, no allocations:
weights live in flash as const arrays, state is a single struct.

  python3 robot/export_c.py --circuit data/fly_circuit_256.json \\
      --eval data/robot_eval.json --condition real --out firmware/flynet.h

The header contains:
  - synapses (pre, post, synapse-count weight, neuron sign) — real FlyWire v783;
  - sensory and motor index lists;
  - the trained readout (mu, sd, W) from descending neurons to two wheels;
  - flynet_init / flynet_step / flynet_wheels — the LIF loop, same as flybrain.py.

Memory for the 256-neuron circuit: ~8 KB flash, ~4 KB RAM (float32).
For 1024 neurons: ~30 KB flash, ~13 KB RAM. A plain Cortex-M4 handles it;
for less, take the 128/256 circuit and squeeze float32 down to int8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CIRCUIT = os.path.join(ROOT, "data", "fly_circuit.json")

TEMPLATE = """/* flynet.h — сгенерировано robot/export_c.py из коннектома FlyWire v783.
 *
 * Контур: {circuit}
 * Источник синапсов: 2025_Connectivity_783.parquet (настоящие связи, не синтетика)
 * Нейронов: {n} · синапсов: {syn} · входов читателя: {nfeat}
 * Fidelity «сенсор → нисходящие нейроны»: {fidelity}
 * Читатель: {kind} · обучено чисел: {nparams}; всё остальное взято из мозга как есть.
 *
 * Использование:
 *     static flynet_t brain;
 *     flynet_init(&brain);
 *     ...
 *     flynet_set_input(&brain, odor_left, odor_right, loom_left, loom_right);
 *     for (int i = 0; i < 10; i++) flynet_step(&brain);   // 10 шагов × 5 мс = 50 мс
 *     float left, right;
 *     flynet_wheels(&brain, &left, &right);               // [-1, 1] на моторы
 *
 * Шаг мозга 5 мс. Проводка и веса — из коннектома, менять их не нужно:
 * обучается только читатель (он в конце файла).
 */
#ifndef FLYNET_H
#define FLYNET_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define FLYNET_N      {n}
#define FLYNET_SYN    {syn}
#define FLYNET_FEAT   {nfeat}
#define FLYNET_DT     5.0f        /* мс */
#define FLYNET_STEPS  10          /* шагов на такт управления (50 мс) */

#define FLY_V_REST    (-52.0f)
#define FLY_V_TH      (-45.0f)
#define FLY_V_RESET   (-54.0f)
#define FLY_TAU_M     20.0f
#define FLY_REFRACT   2.2f
#define FLY_TAU_SYN   5.0f
#define FLY_TAU_RATE  120.0f
#define FLY_I_SCALE   25.0f
#define FLY_GAIN_ODOR {gain_odor}f
#define FLY_GAIN_LOOM {gain_loom}f
#define FLY_RATE_SCALE ((FLYNET_DT / FLY_TAU_RATE) * (1000.0f / FLYNET_DT))

/* ── синапсы: pre[i] → post[i], вес в синапсах, знак берётся у pre ── */
static const int32_t fly_pre[FLYNET_SYN] = {{
{pre}
}};

static const int32_t fly_post[FLYNET_SYN] = {{
{post}
}};

static const int16_t fly_w[FLYNET_SYN] = {{
{w}
}};

static const int8_t fly_sign[FLYNET_N] = {{
{sign}
}};

/* полный синаптический вход клетки — знаменатель нормировки */
static const float fly_in_sum[FLYNET_N] = {{
{in_sum}
}};

/* ── сенсорика: настоящие ORN (запах) и LPLC2/LC4 (наезд), разбитые по сторонам ── */
static const int32_t fly_sens_odor_l[{n_odor_l}] = {{{odor_l}}};
static const int32_t fly_sens_odor_r[{n_odor_r}] = {{{odor_r}}};
static const int32_t fly_sens_loom_l[{n_loom_l}] = {{{loom_l}}};
static const int32_t fly_sens_loom_r[{n_loom_r}] = {{{loom_r}}};

/* ── моторный выход: нисходящие нейроны, с которых читает политика ── */
static const int32_t fly_feat[FLYNET_FEAT] = {{
{feat}
}};

/* ── обученный читатель: (rate − mu) / sd → два колеса ── */
static const float fly_mu[FLYNET_FEAT] = {{
{mu}
}};
static const float fly_sd[FLYNET_FEAT] = {{
{sd}
}};
static const float fly_wout[2][FLYNET_FEAT + 1] = {{
      {{
{wout0}
      }},
      {{
{wout1}
      }}
}};

typedef struct {{
    float V[FLYNET_N];
    float ref[FLYNET_N];
    float Isyn[FLYNET_N];
    float rate[FLYNET_N];
    float ext[FLYNET_N];
    uint8_t spike[FLYNET_N];      /* спайки предыдущего шага — ими и передаётся сигнал */
    float drive[FLYNET_N];
}} flynet_t;

static inline void flynet_init(flynet_t *b) {{
    for (int i = 0; i < FLYNET_N; i++) {{
        b->V[i] = FLY_V_REST;
        b->ref[i] = 0.0f;
        b->Isyn[i] = 0.0f;
        b->rate[i] = 0.0f;
        b->ext[i] = 0.0f;
        b->spike[i] = 0;
        b->drive[i] = 0.0f;
    }}
}}

static inline void flynet_set_input(flynet_t *b, float odor_left, float odor_right,
                                    float loom_left, float loom_right) {{
    memset(b->ext, 0, sizeof(b->ext));
    for (unsigned i = 0; i < sizeof(fly_sens_odor_l) / sizeof(int32_t); i++)
        b->ext[fly_sens_odor_l[i]] += odor_left * FLY_GAIN_ODOR;
    for (unsigned i = 0; i < sizeof(fly_sens_odor_r) / sizeof(int32_t); i++)
        b->ext[fly_sens_odor_r[i]] += odor_right * FLY_GAIN_ODOR;
    for (unsigned i = 0; i < sizeof(fly_sens_loom_l) / sizeof(int32_t); i++)
        b->ext[fly_sens_loom_l[i]] += loom_left * FLY_GAIN_LOOM;
    for (unsigned i = 0; i < sizeof(fly_sens_loom_r) / sizeof(int32_t); i++)
        b->ext[fly_sens_loom_r[i]] += loom_right * FLY_GAIN_LOOM;
}}

/* Один шаг LIF: та же динамика и тот же порядок, что в robot/flybrain.py */
static inline void flynet_step(flynet_t *b) {{
    const float decay_syn = expf(-FLYNET_DT / FLY_TAU_SYN);
    const float decay_rate = expf(-FLYNET_DT / FLY_TAU_RATE);
    memset(b->drive, 0, sizeof(b->drive));

    /* 1. драйв = доля входа клетки, пришедшая от спайков ПРЕДЫДУЩЕГО шага */
    for (int i = 0; i < FLYNET_SYN; i++) {{
        int32_t pre = fly_pre[i];
        if (b->spike[pre])
            b->drive[fly_post[i]] += (float)fly_sign[pre] * (float)fly_w[i] / fly_in_sum[fly_post[i]];
    }}

    /* 2. утечка синапсов, мембрана, порог, рефрактерность, окно чтения */
    for (int i = 0; i < FLYNET_N; i++) {{
        b->Isyn[i] = b->Isyn[i] * decay_syn + b->drive[i];
        b->spike[i] = 0;

        if (b->ref[i] > 0.0f) {{
            b->ref[i] -= FLYNET_DT;
            b->V[i] = FLY_V_RESET;
            b->rate[i] *= decay_rate;
            continue;
        }}
        b->V[i] += (-(b->V[i] - FLY_V_REST) + b->Isyn[i] * FLY_I_SCALE + b->ext[i])
                   * (FLYNET_DT / FLY_TAU_M);
        b->rate[i] *= decay_rate;
        if (b->V[i] >= FLY_V_TH) {{
            b->V[i] = FLY_V_RESET;
            b->ref[i] = FLY_REFRACT;
            b->spike[i] = 1;
            b->rate[i] += FLY_RATE_SCALE;
        }}
    }}
}}

static inline void flynet_wheels(const flynet_t *b, float *left, float *right) {{
    float o0 = fly_wout[0][FLYNET_FEAT], o1 = fly_wout[1][FLYNET_FEAT];
    for (int j = 0; j < FLYNET_FEAT; j++) {{
        float x = (b->rate[fly_feat[j]] - fly_mu[j]) / fly_sd[j];
        o0 += fly_wout[0][j] * x;
        o1 += fly_wout[1][j] * x;
    }}
    if (o0 > 1.0f) o0 = 1.0f; else if (o0 < -1.0f) o0 = -1.0f;
    if (o1 > 1.0f) o1 = 1.0f; else if (o1 < -1.0f) o1 = -1.0f;
    *left = o0; *right = o1;
}}

/* Размеры для проверки на этапе сборки */
#define FLYNET_FLASH_BYTES (sizeof(fly_pre) + sizeof(fly_post) + sizeof(fly_w) \\
    + sizeof(fly_sign) + sizeof(fly_in_sum) + sizeof(fly_mu) + sizeof(fly_sd) \\
    + sizeof(fly_wout) + 4 * (sizeof(fly_sens_odor_l) + sizeof(fly_sens_odor_r) \\
    + sizeof(fly_sens_loom_l) + sizeof(fly_sens_loom_r)) + sizeof(fly_feat))
#define FLYNET_RAM_BYTES   (sizeof(flynet_t))   /* состояние = вся память мозга */

#endif /* FLYNET_H */
"""


def fmt_float_array(values, per_line=6, indent="    "):
    out, line = [], []
    for i, v in enumerate(values):
        s = f"{float(v):.6g}"
        if not any(c in s for c in ".eE"):
            s += ".0"          # 101f — не литерал C, нужно 101.0f
        line.append(s + "f")
        if len(line) == per_line:
            out.append(indent + ", ".join(line) + ",")
            line = []
    if line:
        out.append(indent + ", ".join(line) + ",")
    return "\n".join(out)


def fmt_int_array(values, per_line=14, indent="    "):
    values = list(values)
    if not values:
        return "0"
    out, line = [], []
    for v in values:
        line.append(str(int(v)))
        if len(line) == per_line:
            out.append(indent + ", ".join(line) + ",")
            line = []
    if line:
        out.append(indent + ", ".join(line) + ",")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default=DEFAULT_CIRCUIT)
    ap.add_argument("--eval", dest="eval_path", default="",
                    help="json из robot/eval.py — оттуда берётся обученный читатель")
    ap.add_argument("--condition", default="real", help="какое условие взять из eval")
    ap.add_argument("--out", default=os.path.join(ROOT, "firmware", "flynet.h"))
    args = ap.parse_args()

    with open(args.circuit, encoding="utf-8") as fh:
        circuit = json.load(fh)
    neurons = circuit["neurons"]
    groups = circuit["groups"]
    n = len(neurons)
    sign = np.array([x["sign"] for x in neurons], dtype=np.int8)

    edges = np.asarray(circuit["edges"], dtype=np.float64).reshape(-1, 4)
    pre = edges[:, 0].astype(np.int64)
    post = edges[:, 1].astype(np.int64)
    w = edges[:, 2]
    in_sum = np.bincount(post, weights=np.abs(w), minlength=n).astype(np.float32)
    in_sum[in_sum == 0] = 1.0

    gain_odor = gain_loom = 40.0
    readout = None
    if args.eval_path:
        with open(args.eval_path, encoding="utf-8") as fh:
            ev = json.load(fh)
        entry = None
        for path, rows in ev.get("circuits", {}).items():
            if os.path.basename(path) == os.path.basename(args.circuit):
                entry = rows.get(args.condition)
        if entry is None:
            sys.exit(f"в {args.eval_path} нет условия {args.condition} для {args.circuit}")
        readout = entry["readout"]
        gain_odor = gain_loom = entry["gain"]

    # Кто именно попадает в читатель: только нисходящие нейроны (биология) или всё
    # состояние мозга (--readout all). Определяем по размеру вектора в eval-файле.
    dn = np.concatenate([groups[g] for g in ("turn", "fwd", "bwd") if g in groups])
    kind = "dn"
    if readout is not None and int(readout["feat"]) == n:
        kind = "all"
    feat = np.arange(n, dtype=np.int64) if kind == "all" else dn
    nfeat = len(feat)
    if readout is None:
        mu = np.zeros(nfeat, dtype=np.float32)
        sd = np.ones(nfeat, dtype=np.float32)
        W = np.zeros((2, nfeat + 1), dtype=np.float32)
        print("внимание: читатель не задан (--eval), выгружены нулевые веса")
    else:
        mu = np.asarray(readout["mu"], dtype=np.float32)
        sd = np.asarray(readout["sd"], dtype=np.float32)
        W = np.asarray(readout["W"], dtype=np.float32)
        if W.shape != (2, nfeat + 1):
            sys.exit(f"читатель не сходится с контуром: W {W.shape}, ожидалось (2, {nfeat + 1}). "
                     f"Контур на 1024 нейрона и контуры других размеров имеют разный DN-выход.")

    left_idx = lambda key: groups.get(key, [])          # noqa: E731
    sides = {}
    for key in ("odor", "loom"):
        idx = groups.get(key, [])
        # сторона берётся из аннотаций, она уже внутри контура
        left = [i for i in idx if _side(neurons[i]) == "left"]
        right = [i for i in idx if _side(neurons[i]) == "right"]
        sides[key] = (left, right)

    text = TEMPLATE.format(
        circuit=os.path.basename(args.circuit),
        n=n, syn=len(pre), nfeat=nfeat,
        fidelity=circuit.get("meta", {}).get("fidelity"),
        kind=("всё состояние мозга" if kind == "all" else "только 12 нисходящих нейронов"),
        nparams=2 * (nfeat + 1),
        gain_odor=gain_odor, gain_loom=gain_loom,
        pre=fmt_int_array(pre), post=fmt_int_array(post),
        w=fmt_int_array(w, per_line=20), sign=fmt_int_array(sign, per_line=20),
        in_sum=fmt_float_array(in_sum),
        n_odor_l=len(sides["odor"][0]), odor_l=fmt_int_array(sides["odor"][0], per_line=12),
        n_odor_r=len(sides["odor"][1]), odor_r=fmt_int_array(sides["odor"][1], per_line=12),
        n_loom_l=len(sides["loom"][0]), loom_l=fmt_int_array(sides["loom"][0], per_line=12),
        n_loom_r=len(sides["loom"][1]), loom_r=fmt_int_array(sides["loom"][1], per_line=12),
        feat=fmt_int_array(feat, per_line=12),
        mu=fmt_float_array(mu), sd=fmt_float_array(sd),
        wout0=fmt_float_array(W[0], indent="        "),
        wout1=fmt_float_array(W[1], indent="        "),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"записан {args.out}: {n} нейронов, {len(pre):,} синапсов, "
          f"читатель {2 * (nfeat + 1)} чисел, вход ×{gain_odor:g}")
    print(f"flash ≈ {os.path.getsize(args.out) / 1024:.0f} КБ исходником; "
          f"RAM ≈ {5 * n * 4 / 1024:.1f} КБ (float32 на 5 массивов состояния)")


def _side(neuron) -> str:
    return str(neuron.get("side", ""))


if __name__ == "__main__":
    main()
