"""Revisión de fuga de datos entre train/val/test.

Detecta (a) duplicados exactos (MD5), (b) casi-duplicados (dHash 64 bit, distancia de Hamming <= N)
y (c) imágenes del mismo original (nombre antes de '.rf.') repartidas en splits distintos.
    python ml/check_leakage.py [--dataset datasets/waste] [--hamming 4]
Escribe ml/reports/leakage_report.md y ml/reports/leakage_pairs.csv
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        im.draft("L", (64, 64))
        g = im.convert("L").resize((9, 8), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/waste")
    ap.add_argument("--hamming", type=int, default=4)
    a = ap.parse_args()
    ds = ROOT / a.dataset
    items = []  # (split, path)
    for s in SPLITS:
        items += [(s, p) for p in sorted((ds / "images" / s).glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    print(f"{len(items)} imágenes")

    md5s, hashes = {}, []
    for i, (s, p) in enumerate(items):
        md5s[i] = hashlib.md5(p.read_bytes()).hexdigest()
        hashes.append(dhash(p))
        if i % 3000 == 0:
            print(i, flush=True)

    # (a) exactos
    by_md5 = defaultdict(list)
    for i, h in md5s.items():
        by_md5[h].append(i)
    exact = [g for g in by_md5.values() if len({items[i][0] for i in g}) > 1]

    # (c) mismo original en splits distintos
    by_base = defaultdict(set)
    for s, p in items:
        by_base[p.stem.split(".rf.")[0]].add(s)
    same_base = {k: v for k, v in by_base.items() if len(v) > 1}

    # (b) casi-duplicados: 5 trozos de ~13 bits; si distancia <= 4, al menos un trozo coincide (principio del palomar)
    chunks = [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]
    buckets = [defaultdict(list) for _ in chunks]
    for i, h in enumerate(hashes):
        for k, (off, w) in enumerate(chunks):
            buckets[k][(h >> off) & ((1 << w) - 1)].append(i)
    pairs = set()
    for b in buckets:
        for idx in b.values():
            if 1 < len(idx) <= 400:
                for i, j in combinations(idx, 2):
                    if items[i][0] != items[j][0] and bin(hashes[i] ^ hashes[j]).count("1") <= a.hamming:
                        pairs.add((min(i, j), max(i, j)))
    cross = defaultdict(int)
    for i, j in pairs:
        cross["/".join(sorted((items[i][0], items[j][0])))] += 1
    leaked_val_test = {j if items[j][0] != "train" else i for i, j in pairs if "train" in (items[i][0], items[j][0])}

    out = ROOT / "ml" / "reports"
    out.mkdir(exist_ok=True)
    with open(out / "leakage_pairs.csv", "w") as f:
        f.write("split_a,file_a,split_b,file_b,hamming\n")
        for i, j in sorted(pairs):
            f.write(f"{items[i][0]},{items[i][1].name},{items[j][0]},{items[j][1].name},{bin(hashes[i]^hashes[j]).count('1')}\n")
    n = {s: sum(1 for x in items if x[0] == s) for s in SPLITS}
    md = [
        "# Revisión de fuga de datos", "",
        f"Dataset: `{a.dataset}` · imágenes: " + ", ".join(f"{s}={n[s]}" for s in SPLITS), "",
        f"- Duplicados exactos (MD5) en splits distintos: **{len(exact)}** grupos",
        f"- Mismo original (antes de `.rf.`) en splits distintos: **{len(same_base)}**",
        f"- Casi-duplicados (dHash, Hamming ≤ {a.hamming}) entre splits distintos: **{len(pairs)}** pares " + (str(dict(cross)) if cross else ""),
        f"- Imágenes de val/test con un casi-duplicado en train: **{len(leaked_val_test)}** "
        f"({100*len(leaked_val_test)/max(1,n['val']+n['test']):.1f} % de val+test)", "",
        "Pares completos en `leakage_pairs.csv`. Los casi-duplicados son una cota superior (ver nota de verificación manual al final del informe).",
    ]
    (out / "leakage_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
