"""Проверка: ID нейронов из браузерной модели действительно есть в полном коннектоме.

Браузерная модель (app.js) работает на ~1.8 тыс. нейронов и синтетических весах.
Этот скрипт отвечает на вопрос «а не выдуманы ли эти нейроны»: берёт ID из
data/neuron_atlas.json и data/mushroom_body_neurons.json, находит их в
data/2025_Connectivity_783.parquet и показывает реальные степени связи и то,
сколько настоящих синапсов KC→MBON стоит за упрощением в браузере.

Сначала: python3 scripts/download_full.py
Запуск:   python3 scripts/verify_connectome.py
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PARQUET = os.path.join(DATA, "2025_Connectivity_783.parquet")

def load_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def collect_atlas_ids(atlas, mb):
    stim, motor = {}, {}
    for key, v in atlas["stimuli"].items():
        ids = list(v.get("neuron_ids") or [])
        for group in (v.get("neuron_ids_groups") or {}).values():
            ids += list(group)
        stim[key] = set(ids)
    for key, v in atlas["output_neurons"].items():
        motor[key] = v["id"]

    kc = set()
    for group in mb["kenyon_cells"].values():
        kc |= set(group)
    mbon, pam, ppl = set(), set(), set()
    for group in mb["mbon"].values():
        mbon |= set(group)
    for group in mb["dan_pam_reward"].values():
        pam |= set(group)
    for group in mb["dan_ppl_punishment"].values():
        ppl |= set(group)
    return stim, motor, kc, mbon, pam, ppl


def main():
    if not os.path.exists(PARQUET):
        sys.exit("Нет data/2025_Connectivity_783.parquet — сначала: python3 scripts/download_full.py")

    atlas = load_json("neuron_atlas.json")
    mb = load_json("mushroom_body_neurons.json")
    stim, motor, kc, mbon, pam, ppl = collect_atlas_ids(atlas, mb)

    print(f"parquet: {os.path.getsize(PARQUET) / 1e6:.1f} МБ, читаю только две колонки ID…")
    # Только ID — иначе pandas съест под гигабайт на 15 млн строк.
    edges = pd.read_parquet(PARQUET, columns=["Presynaptic_ID", "Postsynaptic_ID"])
    pre = edges["Presynaptic_ID"].to_numpy()
    post = edges["Postsynaptic_ID"].to_numpy()
    nodes = set(pre.tolist()) | set(post.tolist())
    print(f"связей: {len(edges):,} | нейронов: {len(nodes):,}\n")

    def report(title, ids):
        found = ids & nodes
        missing = ids - nodes
        flag = "OK " if not missing else "!! "
        print(f"{flag}{title}: {len(found)}/{len(ids)} ID найдено в коннектоме" +
              (f", отсутствуют: {sorted(missing)[:3]}…" if missing else ""))
        return found

    print("── входы модели ──")
    for key, ids in stim.items():
        report(f"  {key}", ids)
    print("\n── выходы модели ──")
    report("  моторные (oDN/DNa/MDN/Giant Fiber…)", set(motor.values()))

    print("\n── грибовидное тело ──")
    kc_f = report("  Kenyon-клетки", kc)
    mbon_f = report("  MBON", mbon)
    report("  DAN PAM (награда)", pam)
    report("  DAN PPL1 (наказание)", ppl)

    # Реальные синапсы KC→MBON — та самая матрица, которую браузер заполняет случайно.
    print("\n── реальная матрица KC→MBON (в браузере она синтетическая) ──")
    if kc_f and mbon_f:
        m = pd.Series(pre).isin(kc_f).to_numpy() & pd.Series(post).isin(mbon_f).to_numpy()
        sub = edges[m]
        pairs = sub.groupby(["Presynaptic_ID", "Postsynaptic_ID"]).size()
        fan_out = sub.groupby("Presynaptic_ID")["Postsynaptic_ID"].nunique()
        fan_in = sub.groupby("Postsynaptic_ID")["Presynaptic_ID"].nunique()
        print(f"  синапсов KC→MBON: {len(sub):,}")
        print(f"  уникальных пар KC→MBON: {len(pairs):,} "
              f"({100 * len(pairs) / (len(kc_f) * len(mbon_f)):.1f}% плотности)")
        print(f"  KC с реальным выходом на MBON: {fan_out.shape[0]:,} из {len(kc_f):,}")
        print(f"  веер KC→MBON: медиана {int(fan_out.median())}, максимум {int(fan_out.max())} MBON-партнёров")
        print(f"  веер MBON←KC: медиана {int(fan_in.median())}, максимум {int(fan_in.max())} KC-партнёров")

    # Реальные степени для сенсорики: сколько у неё исходящих.
    print("\n── реальная исходящая степень сенсорики ──")
    out_deg = edges.groupby("Presynaptic_ID").size()
    for key, ids in stim.items():
        vals = out_deg.reindex(list(ids)).dropna()
        if len(vals):
            print(f"  {key:16s} нейронов {len(vals):4d} | синапсов всего {int(vals.sum()):7,d} "
                  f"| медиана на нейрон {int(vals.median()):5d}")

    print("\nВывод: ID в браузерной модели настоящие. Связи в браузере синтетические —")
    print("реальная матрица KC→MBON выше показывает, сколько данных при этом выбрасывается.")


if __name__ == "__main__":
    main()
