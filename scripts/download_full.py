"""Скачать ПОЛНЫЙ коннектом FlyWire v783 в data/ (parquet ~96 МБ + аннотации ~31 МБ).

Зачем он нужен:
  Браузеру он НЕ нужен — index.html читает только маленькие data/neuron_atlas.json
  и data/mushroom_body_neurons.json. Полный parquet (~5 млн синапсов) нужен для
  Python-симуляции всего мозга (Brian2/PyTorch) и для проверки модели по реальным связям.

Почему не raw.githubusercontent:
  Файлы лежат в git-lfs, и по обычной ссылке отдаётся 134-байтная заглушка-указатель.
  Скрипт читает указатель, берёт из него ожидаемый size + sha256 и качает настоящий
  файл с media.githubusercontent.com, докачивая по Range при обрыве.

Запуск:
  python3 scripts/download_full.py            # скачать (с докачкой)
  python3 scripts/download_full.py --check    # только проверить, что уже лежит
"""
import hashlib
import os
import sys
import urllib.error
import urllib.request

REPO = "lixiang1076/fly-brain"
BRANCH = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/data"
FILES = ["2025_Connectivity_783.parquet", "flywire_annotations.tsv"]
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHUNK = 1 << 20


def human(n):
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.1f} {unit}" if unit != "Б" else f"{n} Б"
        n /= 1024.0


def read_pointer(name):
    """Читаем LFS-указатель: он маленький и содержит size + sha256."""
    try:
        with urllib.request.urlopen(f"{RAW}/{name}", timeout=30) as r:
            txt = r.read(4096).decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise SystemExit(f"Не достучались до {RAW}/{name}: {e}")
    if not txt.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    meta = {}
    for line in txt.strip().splitlines()[1:]:
        if " " in line:
            k, v = line.split(" ", 1)
            meta[k.strip()] = v.strip()
    oid = meta.get("oid", "")
    return (oid.split(":")[-1] or None, int(meta.get("size", 0) or 0))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(name, size, sha):
    out = os.path.join(DATA, name)
    url = f"{MEDIA}/{name}"
    part = out + ".part"

    done = os.path.getsize(part) if os.path.exists(part) else 0
    if done > size:
        os.remove(part)
        done = 0

    req = urllib.request.Request(url)
    if done:
        req.add_header("Range", f"bytes={done}-")
        print(f"  докачиваем с {human(done)} из {human(size)}")
    mode = "ab" if done else "wb"

    last_shown = -1
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, mode) as f:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = int(done * 100 / size) if size else 0
                if pct // 5 != last_shown // 5:
                    last_shown = pct
                    print(f"  {pct:3d}%  {human(done)} / {human(size)}", flush=True)
    except urllib.error.URLError as e:
        raise SystemExit(f"Обрыв на {human(done)}. Запусти ещё раз — докачает с этого места. ({e})")

    if size and done != size:
        raise SystemExit(f"Размер не совпал: получили {done}, ждали {size}. Файл оставлен как {part} для докачки.")

    got = sha256_of(part)
    if sha and got != sha:
        os.remove(part)
        raise SystemExit(f"sha256 не совпал (получили {got[:16]}…, ждали {sha[:16]}…) — файл удалён, качай заново.")

    os.replace(part, out)
    return out


def verify_parquet(path):
    """Parquet начинается и заканчивается магией PAR1 — быстрая проверка, что это не заглушка."""
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        tail = f.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def already_ok(name, size, sha):
    out = os.path.join(DATA, name)
    if not os.path.exists(out) or (size and os.path.getsize(out) != size):
        return False
    if sha and sha256_of(out) != sha:
        return False
    return True


def main():
    check_only = "--check" in sys.argv
    os.makedirs(DATA, exist_ok=True)
    print(f"Папка данных: {DATA}\n")

    for name in FILES:
        print(f"→ {name}")
        meta = read_pointer(name)
        if meta is None:
            raise SystemExit(f"  {RAW}/{name} отдаёт не LFS-указатель. Проверь ссылку вручную.")
        sha, size = meta
        print(f"  ожидается {human(size)}" + (f", sha256 {sha[:16]}…" if sha else ""))

        if meta and already_ok(name, size, sha):
            print("  уже на диске и совпадает по sha256 — пропускаю")
            continue

        if check_only:
            print("  НЕТ на диске (или не совпадает)")
            continue

        path = download(name, size, sha)
        print(f"  готово: {path} ({human(os.path.getsize(path))})")
        if name.endswith(".parquet"):
            print("  parquet magic bytes:", "OK" if verify_parquet(path) else "ПЛОХО")

    print("\nИтог:")
    for name in FILES:
        p = os.path.join(DATA, name)
        print(f"  {'OK ' if os.path.exists(p) else '-- '}{name}" + (f"  {human(os.path.getsize(p))}" if os.path.exists(p) else ""))
    print("\nЭто данные для Python-симуляции всего мозга. Браузер их не читает —")
    print("ему достаточно neuron_atlas.json и mushroom_body_neurons.json.")


if __name__ == "__main__":
    main()
