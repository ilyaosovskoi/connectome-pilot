"""extract_circuit.py — cut a working sensorimotor circuit for the robot out of
the FULL FlyWire v783 connectome: sensors -> interneurons -> descending neurons.

Why. The browser model (app.js) takes only neuron IDs from the connectome and
draws random connections. Fine for a game, not for a robot: a robot needs real
weights, otherwise there was no point counting 15 million synapses.

What the script does:

  1. reads data/2025_Connectivity_783.parquet (15,091,983 synapses) and
     data/flywire_annotations.tsv (cell type, transmitter, side, soma);
  2. scores flow through each of the 139,244 cells between input and output:
     personalized PageRank forward from sensors (f) and backward from
     descending neurons (g), score = f * g. This measures how much a cell lies
     on sensor -> DN paths, not just how "popular" it is;
  3. keeps a budget of N top-scoring neurons (+ mandatory inputs/outputs) and
     builds the INDUCED subgraph — only real synapses between kept cells;
  4. measures fidelity: descending-neuron responses of the circuit vs the full
     brain on the same input (Pearson correlation + L1 error);
  5. writes data/fly_circuit.json — a compact circuit for the runtime
     (robot/flybrain.py) with types, sides, signs and weights.

Run:
  python3 scripts/extract_circuit.py                       # budget 1024, default groups
  python3 scripts/extract_circuit.py --budget 256 --out data/fly_circuit_256.json
  python3 scripts/extract_circuit.py --sweep               # fidelity(N) curve
  python3 scripts/extract_circuit.py --sensors odor,loom   # only these modalities

Requires: numpy, pandas, pyarrow, scipy. Fetch data via scripts/download_full.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PARQUET = os.path.join(DATA, "2025_Connectivity_783.parquet")
ANNOTATIONS = os.path.join(DATA, "flywire_annotations.tsv")

# ──────────────────────────── входы и выходы контура ────────────────────────────
#
# Сенсорные группы — это то, что у робота реально есть: обоняние и «зрение»
# (у мушки близость/тень ловят LC4/LPLC2, у робота это дальномер или ИК).
# Регулярки по cell_type, как они записаны в аннотациях FlyWire.

SENSOR_GROUPS = {
    "odor": [r"^ORN_"],                      # обонятельные рецепторные нейроны, 2 282
    "loom": [r"^LPLC2$", r"^LC4", r"^LC45", r"^LC46"],   # тень / наезд, ~570
    "motion": [r"^T4[abcd]$", r"^T5[abcd]$"],            # движение, 12 246
    "touch": [r"^JO-"],                      # механосенсорика (джонстонов орган)
    "sugar": [r"^BM_Taste"],                 # вкус
}

# Нисходящие нейроны — это единственный моторный выход мозга. У робота они
# становятся колёсами: DNa01/DNa02 — поворот, DNp09 — скорость вперёд, MDN — назад.
OUTPUT_GROUPS = {
    "turn": [r"^DNa01$", r"^DNa02$"],
    "fwd": [r"^DNp09$"],
    "bwd": [r"^MDN$", r"^DNp01$"],
}

# Нейромедиатор → знак связи. Ацетилхолин в этой сети возбуждающий, ГАМК и
# глутамат — тормозные. Дофамин/серотонин/октопамин передаём как модуляцию,
# для рантайма считаем их знак положительным (иначе LIF уходит в шум).
INHIBITORY_NT = {"gaba", "glutamate"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ──────────────────────────────── чтение данных ────────────────────────────────

def load_annotations() -> pd.DataFrame:
    if not os.path.exists(ANNOTATIONS):
        sys.exit(f"нет {ANNOTATIONS} — скачай данные: python3 scripts/download_full.py")
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann = ann[ann["root_id"].notna()].drop_duplicates("root_id").reset_index(drop=True)
    for col in ("cell_type", "super_class", "side", "top_nt"):
        ann[col] = ann[col].fillna("").astype(str)
    return ann


def match_group(ann: pd.DataFrame, patterns: list[str]) -> np.ndarray:
    """Индексы строк аннотации, чей cell_type попадает под любую из регулярок."""
    if not patterns:
        return np.array([], dtype=np.int64)
    rx = re.compile("|".join(f"(?:{p})" for p in patterns))
    mask = ann["cell_type"].map(lambda t: bool(rx.match(t)))
    return np.nonzero(mask.to_numpy())[0].astype(np.int64)


def build_graph(ann: pd.DataFrame):
    """CSR-матрица синапсов: W[pre, post] = число синапсов (знак отдельно)."""
    ids = np.sort(ann["root_id"].to_numpy(np.int64))
    n = len(ids)
    pf = pq.ParquetFile(PARQUET)
    log(f"parquet: {os.path.getsize(PARQUET) / 1e6:.0f} МБ, "
        f"{pf.metadata.num_rows:,} связей, {pf.metadata.num_row_groups} row group")

    pre_parts, post_parts, w_parts = [], [], []
    dropped = 0
    t0 = time.time()
    for rg in range(pf.metadata.num_row_groups):
        tb = pf.read_row_group(rg, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
        pre = tb.column("Presynaptic_ID").to_numpy()
        post = tb.column("Postsynaptic_ID").to_numpy()
        w = tb.column("Connectivity").to_numpy()
        pi = np.searchsorted(ids, pre)
        po = np.searchsorted(ids, post)
        # searchsorted может дать n — обрезаем и потом выбраковываем по равенству
        np.clip(pi, 0, n - 1, out=pi)
        np.clip(po, 0, n - 1, out=po)
        ok = (ids[pi] == pre) & (ids[po] == post)
        dropped += int((~ok).sum())
        pre_parts.append(pi[ok].astype(np.int32))
        post_parts.append(po[ok].astype(np.int32))
        w_parts.append(w[ok].astype(np.int32))

    pre = np.concatenate(pre_parts)
    post = np.concatenate(post_parts)
    w = np.concatenate(w_parts)
    log(f"граф собран за {time.time() - t0:.1f} с: {len(pre):,} связей, "
        f"отброшено {dropped:,} (нет в аннотациях)")

    W = sp.coo_matrix((w.astype(np.float32), (pre, post)), shape=(n, n)).tocsr()
    W.sum_duplicates()
    log(f"уникальных пар pre→post: {W.nnz:,} (синапсов {int(W.sum()):,})")
    return ids, W, pre, post, w


# ──────────────────────────────── «ток» через узел ────────────────────────────────

def personalized_pagerank(P_T, seed_vec, alpha: float = 0.85, iters: int = 80):
    """f = alpha · P^T f + (1 - alpha) · seed — стационарный поток от seed-узлов."""
    f = seed_vec.copy()
    for _ in range(iters):
        f = alpha * (P_T @ f) + (1.0 - alpha) * seed_vec
    return f


def propagate(P_T: sp.csr_matrix, seeds: np.ndarray, alpha: float = 0.85,
              iters: int = 80) -> np.ndarray:
    """Столбцы — отдельные сенсорные группы, строки — нейроны. F = alpha·P^T·F + (1-alpha)·S.

    Входы считаем по группам отдельно: тогда видно не только «сколько сигнала
    дожило», но и не перепуталось ли, какой сенсор чем командует.
    """
    F = seeds.astype(np.float32).copy()
    for _ in range(iters):
        F = alpha * (P_T @ F) + (1.0 - alpha) * seeds
    return F


def group_seeds(members: list[np.ndarray], n: int) -> np.ndarray:
    """Матрица seed-векторов (n × кол-во групп), каждый столбец нормирован."""
    S = np.zeros((n, len(members)), dtype=np.float32)
    for j, rows in enumerate(members):
        if len(rows) == 0:
            continue
        S[rows, j] = 1.0
        S[:, j] /= S[:, j].sum()
    return S


def flow_scores(W: sp.csr_matrix, sources: np.ndarray, targets: np.ndarray,
                alpha: float = 0.85, iters: int = 80):
    """score = f · g — сколько сигнала проходит через узел по пути вход → выход.

    Структурный поток считаем на |весах|: ингибирование тоже часть пути,
    нам важна топология информационного канала, а не знак отдельного синапса.
    """
    out_sum = np.asarray(W.sum(axis=1)).ravel()
    out_sum[out_sum == 0] = 1.0
    P = W.multiply(1.0 / out_sum[:, None]).tocsr()      # строки: pre → post
    P_T = P.T.tocsr()

    a = np.zeros(W.shape[0], dtype=np.float32)
    a[sources] = 1.0
    a /= max(a.sum(), 1e-9)
    b = np.zeros(W.shape[0], dtype=np.float32)
    b[targets] = 1.0
    b /= max(b.sum(), 1e-9)

    f = personalized_pagerank(P_T, a, alpha, iters)   # вход → узел
    g = personalized_pagerank(P, b, alpha, iters)     # узел → выход
    return f, g, f * g, P, P_T


# ──────────────────────────────── выбор нейронов ────────────────────────────────

def select_nodes(score: np.ndarray, must: np.ndarray, budget: int) -> np.ndarray:
    """Бюджет нейронов: обязательные входы/выходы + лучшие по score.

    Квоты на модальности раздаются заранее — вызывающей стороной через
    pick_group_quota и попадают сюда уже внутри `must`: при малом бюджете иначе
    выкинуло бы, например, всю обонятельную сенсорику — у неё поток ниже, чем
    у зрительной, но без неё робот ничего не чует.
    """
    n = len(score)
    budget = min(budget, n)
    taken = np.zeros(n, dtype=bool)
    taken[must] = True
    need = budget - int(taken.sum())
    if need > 0:
        for idx in np.argsort(-score, kind="stable"):
            if need <= 0:
                break
            if not taken[idx]:
                taken[idx] = True
                need -= 1
    return np.nonzero(taken)[0].astype(np.int64)


def pick_group_quota(score: np.ndarray, members: np.ndarray, quota: int) -> np.ndarray:
    """Топ-quota членов группы по score (гарантированный вход в контур)."""
    if len(members) == 0 or quota <= 0:
        return np.array([], dtype=np.int64)
    k = min(quota, len(members))
    order = members[np.argsort(-score[members], kind="stable")[:k]]
    return order.astype(np.int64)


def sign_by_nt(ann: pd.DataFrame) -> np.ndarray:
    """Знак нейрона по основному нейромедиатору: ГАМК/глутамат — тормозные."""
    sign = np.ones(len(ann), dtype=np.int8)
    sign[ann["top_nt"].isin(INHIBITORY_NT).to_numpy()] = -1
    return sign


def induce(W: sp.csr_matrix, sign: np.ndarray, nodes: np.ndarray,
           max_fanout: int, min_weight: int):
    """Индуцированный подграф + обрезка веера: у мухи тоже не всё со всем связано."""
    keep = np.zeros(W.shape[0], dtype=bool)
    keep[nodes] = True
    sub = (W[keep][:, keep]).tocsr()

    coo = sub.tocoo()
    m = (coo.row != coo.col) & (coo.data >= min_weight)
    r, c, d = coo.row[m], coo.col[m], coo.data[m]

    if max_fanout > 0:
        keep_mask = np.zeros(len(d), dtype=bool)
        order = np.lexsort((-d, r))            # внутри каждого pre — по убыванию веса
        src = r[order]
        rank = np.arange(len(order))
        # номер связи внутри своего pre
        first = np.searchsorted(src, src, side="left")
        pos_in_src = rank - first
        keep_mask[order[pos_in_src < max_fanout]] = True
        r, c, d = r[keep_mask], c[keep_mask], d[keep_mask]

    gsign = sign[r].astype(np.int8)
    return r.astype(np.int32), c.astype(np.int32), d.astype(np.float32), gsign


# ──────────────────────────────── fidelity ────────────────────────────────

def response_fidelity(P_T_full: sp.csr_matrix, P_T_sub: sp.csr_matrix,
                      seeds: np.ndarray, nodes: np.ndarray, targets: np.ndarray,
                      alpha: float = 0.85, iters: int = 80):
    """Насколько матрица «сенсорная группа → нисходящий нейрон» дожила до контура.

    Сравниваем нормированные отклики каждой модальности на каждом выходе.
    Усечённая матрица берётся из полной (нормировки строк не пересчитываются),
    поэтому потерянный на выкинутых клетках поток честно виден как ошибка.
    """
    F_full = propagate(P_T_full, seeds, alpha, iters)
    F_sub = propagate(P_T_sub, seeds[nodes], alpha, iters)

    local = np.full(P_T_full.shape[0], -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    tloc = local[targets]
    keep = tloc >= 0
    if not keep.any():
        return 0.0, 1.0, None, None

    x = F_full[targets[keep]].astype(np.float64)
    y = F_sub[tloc[keep]].astype(np.float64)
    xn = x / np.maximum(np.abs(x).sum(axis=0, keepdims=True), 1e-12)
    yn = y / np.maximum(np.abs(y).sum(axis=0, keepdims=True), 1e-12)
    if xn.std() < 1e-12 or yn.std() < 1e-12:
        return 0.0, 1.0, xn, yn
    corr = float(np.corrcoef(xn.ravel(), yn.ravel())[0, 1])
    err = float(np.abs(xn - yn).sum() / (2.0 * xn.shape[1]))
    return corr, err, xn, yn


# ──────────────────────────────── сборка контура ────────────────────────────────

def extract(ann, ids, W, sensor_patterns: dict, output_patterns: dict,
            budget: int, sensor_quota: int, max_fanout: int, min_weight: int,
            verbose: bool = True):
    n = W.shape[0]
    sensors: dict[str, np.ndarray] = {}
    for name, pats in sensor_patterns.items():
        rows = match_group(ann, pats)
        if len(rows):
            sensors[name] = rows
    outputs: dict[str, np.ndarray] = {}
    for name, pats in output_patterns.items():
        rows = match_group(ann, pats)
        if len(rows):
            outputs[name] = rows

    if not sensors or not outputs:
        sys.exit("не нашёл сенсорные группы или нисходящие нейроны — проверь аннотации")

    if verbose:
        log("\n── входы (сенсорика) ──")
        for k, v in sensors.items():
            log(f"  {k:8s} {len(v):6,d} клеток")
        log("── выходы (нисходящие) ──")
        for k, v in outputs.items():
            log(f"  {k:8s} {len(v):6,d} клеток")

    src = np.concatenate([v for v in sensors.values()])
    tgt = np.concatenate([v for v in outputs.values()])
    seed_matrix = group_seeds(list(sensors.values()), n)
    f, g, score, P, P_T = flow_scores(W, src, tgt)

    # обязательные нейроны: все выходы + топ-квота по каждой модальности
    must = [tgt]
    for name, rows in sensors.items():
        must.append(pick_group_quota(score, rows, sensor_quota))
    must = np.unique(np.concatenate(must))

    nodes = select_nodes(score, must, budget)

    r, c, d, gsign = induce(W, sign_by_nt(ann), nodes, max_fanout, min_weight)
    n_sub = len(nodes)

    # fidelity меряем на УСЕЧЁННОЙ матрице полного мозга: нормировка строк
    # остаётся той же, что у всего мозга, поэтому поток, ушедший в выкинутые
    # клетки, честно теряется. Так метрика растёт монотонно с бюджетом и
    # означает ровно одно: сколько сенсомоторного сигнала дожило до выходов.
    keep = np.zeros(n, dtype=bool)
    keep[nodes] = True
    P_sub = (P[keep][:, keep]).T.tocsr()

    corr, err, xn, yn = response_fidelity(P_T, P_sub, seed_matrix, nodes, tgt)

    if verbose:
        kept_out = len(np.intersect1d(nodes, tgt))
        log(f"\n── контур: {n_sub} нейронов, {len(r):,} связей "
            f"({100 * n_sub / n:.1f}% мозга) ──")
        log(f"  fidelity «сенсор → DN»: r = {corr:.3f}, ошибка L1 = {err:.3f}")
        log(f"  нисходящих нейронов сохранено: {kept_out}/{len(tgt)}")

    return dict(nodes=nodes, r=r, c=c, d=d, gsign=gsign, sensors=sensors,
                outputs=outputs, fidelity=float(corr), l1=float(err),
                P_sub=P_sub, score=score, target_flow_full=xn, target_flow_sub=yn)


def build_export(ann, circuit, meta: dict) -> dict:
    nodes = circuit["nodes"]
    # карта: глобальный индекс → позиция в контуре
    local = np.full(ann.shape[0], -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    sign = sign_by_nt(ann)

    neurons = []
    for idx in nodes:
        row = ann.iloc[idx]
        neurons.append({
            "id": int(row["root_id"]),
            "type": str(row["cell_type"]),
            "cls": str(row["super_class"]),
            "side": str(row["side"]),
            "nt": str(row["top_nt"]),
            "sign": int(sign[idx]),
            "x": round(float(row["soma_x"]) / 1000.0, 1) if pd.notna(row["soma_x"]) else 0.0,
            "y": round(float(row["soma_y"]) / 1000.0, 1) if pd.notna(row["soma_y"]) else 0.0,
            "z": round(float(row["soma_z"]) / 1000.0, 1) if pd.notna(row["soma_z"]) else 0.0,
        })

    groups: dict[str, list[int]] = {}
    for name, rows in circuit["sensors"].items():
        groups[name] = sorted(int(local[x]) for x in rows if local[x] >= 0)
    for name, rows in circuit["outputs"].items():
        groups[name] = sorted(int(local[x]) for x in rows if local[x] >= 0)

    edges = [[int(a_), int(b_), round(float(w), 2), int(s)]
             for a_, b_, w, s in zip(circuit["r"], circuit["c"], circuit["d"], circuit["gsign"])]

    return {
        "meta": meta,
        "groups": groups,
        "neurons": neurons,
        "edges": edges,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=1024, help="сколько нейронов оставить")
    ap.add_argument("--sensors", default="odor,loom",
                    help="модальности входа через запятую: " + ", ".join(SENSOR_GROUPS))
    ap.add_argument("--outputs", default="turn,fwd,bwd",
                    help="моторные группы выхода через запятую: " + ", ".join(OUTPUT_GROUPS))
    ap.add_argument("--sensor-quota", type=int, default=16,
                    help="сколько сенсорных клеток каждой модальности оставить обязательно")
    ap.add_argument("--max-fanout", type=int, default=12, help="максимум исходящих связей на нейрон")
    ap.add_argument("--min-weight", type=int, default=3, help="отбрасывать связи слабее N синапсов")
    ap.add_argument("--out", default=os.path.join(DATA, "fly_circuit.json"))
    ap.add_argument("--sweep", action="store_true", help="показать fidelity(N) и выйти")
    args = ap.parse_args()

    sensor_names = [s.strip() for s in args.sensors.split(",") if s.strip()]
    output_names = [o.strip() for o in args.outputs.split(",") if o.strip()]
    sensor_patterns = {k: SENSOR_GROUPS[k] for k in sensor_names if k in SENSOR_GROUPS}
    output_patterns = {k: OUTPUT_GROUPS[k] for k in output_names if k in OUTPUT_GROUPS}

    log("── аннотации ──")
    ann = load_annotations()
    log(f"клеток: {len(ann):,}")
    ids, W, *_ = build_graph(ann)

    if args.sweep:
        log("\n── fidelity(N): сколько сенсомоторного сигнала доживает ──")
        log(f"{'нейронов':>9} | {'связей':>8} | {'corr':>6} | {'L1':>6}")
        for budget in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
            c = extract(ann, ids, W, sensor_patterns, output_patterns, budget,
                        args.sensor_quota, args.max_fanout, args.min_weight, verbose=False)
            log(f"{budget:>9} | {len(c['r']):>8,} | {c['fidelity']:>6.3f} | {c['l1']:>6.3f}")
        return

    log("\n── извлечение контура ──")
    c = extract(ann, ids, W, sensor_patterns, output_patterns, args.budget,
                args.sensor_quota, args.max_fanout, args.min_weight)

    meta = {
        "source": "FlyWire v783 (Dorkenwald et al. 2024)",
        "budget": args.budget,
        "sensors": sensor_names,
        "outputs": output_names,
        "sensor_quota": args.sensor_quota,
        "max_fanout": args.max_fanout,
        "min_weight": args.min_weight,
        "fidelity": round(c["fidelity"], 4),
        "l1_error": round(c["l1"], 4),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "связи настоящие: все синапсы из 2025_Connectivity_783.parquet",
    }
    export = build_export(ann, c, meta)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(export, fh, ensure_ascii=False)

    log(f"\nзаписан {args.out}: {len(export['neurons'])} нейронов, "
        f"{len(export['edges']):,} связей, {os.path.getsize(args.out) / 1024:.0f} КБ")
    log("группы: " + ", ".join(f"{k}={len(v)}" for k, v in export["groups"].items()))


if __name__ == "__main__":
    main()
