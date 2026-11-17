"""Unifica datasets externos (YOLO/Roboflow, COCO, carpetas) a las 9 clases del proyecto.

    python ml/prepare_dataset.py --config ml/sources.yaml --out datasets/waste
    python ml/prepare_dataset.py --config ml/sources.yaml --dry-run      # solo estadísticas

Salida (formato YOLO):  datasets/waste/images/{train,val,test}  +  labels/{train,val,test}
El reparto train/val/test es determinista (hash del nombre del archivo) para evitar fugas entre ejecuciones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = [c for _, c in sorted(yaml.safe_load((ROOT / "ml" / "data.yaml").read_text())["names"].items())]
CID = {c: i for i, c in enumerate(CLASSES)}
BOTTLES = {"botella_vidrio", "botella_plastico"}


def rp(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_map(path: str) -> dict[str, str | None]:
    m = yaml.safe_load(rp(path).read_text(encoding="utf-8"))["map"]
    bad = {v for v in m.values() if v is not None and v not in CID}
    if bad:
        raise SystemExit(f"[ERROR] {path}: clases destino inválidas {bad}. Válidas: {CLASSES}")
    return {str(k).lower().strip(): v for k, v in m.items()}


def split_of(key: str, ratios=(0.8, 0.1, 0.1)) -> str:
    r = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "train" if r < ratios[0] else "val" if r < ratios[0] + ratios[1] else "test"


# ----------------------------- lectores: devuelven (img_path, [(clase_origen, x1,y1,x2,y2)], (w,h)) -----------------
def read_yolo(src: dict):
    root = rp(src["path"])
    names_yaml = next(root.glob("data.y*ml"), None)
    if not names_yaml:
        raise SystemExit(f"[ERROR] {root}: falta data.yaml")
    names = yaml.safe_load(names_yaml.read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    from PIL import Image

    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT):
        lbl = Path(str(img).replace(f"{'/'}images{'/'}", f"{'/'}labels{'/'}")).with_suffix(".txt")
        if not lbl.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        boxes = []
        for line in lbl.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            cls = names[int(float(t[0]))]
            v = list(map(float, t[1:]))
            if len(v) == 4:
                cx, cy, bw, bh = v
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            else:  # polígono de segmentación -> caja
                xs, ys = v[0::2], v[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append((cls, x1 * w, y1 * h, x2 * w, y2 * h))
        yield img, boxes, (w, h)


def read_coco(src: dict):
    ann = json.loads(rp(src["annotations"]).read_text(encoding="utf-8"))
    cats = {c["id"]: c for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    per: dict[int, list] = {}
    for a in ann["annotations"]:
        c = cats[a["category_id"]]
        x, y, w, h = a["bbox"]
        per.setdefault(a["image_id"], []).append((c["name"], x, y, x + w, y + h, c.get("supercategory", "")))
    root = rp(src.get("images_root", src["path"]))
    for iid, boxes in per.items():
        info = imgs[iid]
        path = root / info["file_name"]
        if path.exists():
            yield path, boxes, (info["width"], info["height"])


def read_folder(src: dict, margin: float = 0.03):
    from PIL import Image

    root = rp(src["path"])
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT):
        with Image.open(img) as im:
            w, h = im.size
        cls = img.parent.name
        yield img, [(cls, w * margin, h * margin, w * (1 - margin), h * (1 - margin))], (w, h)


READERS = {"yolo": read_yolo, "coco": read_coco, "folder": read_folder}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/sources.yaml")
    ap.add_argument("--out", default="datasets/waste")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--symlink", action="store_true", help="enlaza imágenes en vez de copiarlas")
    ap.add_argument("--min-box", type=float, default=2.0, help="descarta cajas con lado < N píxeles")
    ap.add_argument("--bottle-superclass", action="store_true",
                    help="cada botella_vidrio/botella_plastico añade además una caja 'botella' (multi-etiqueta)")
    ap.add_argument("--small-mode", choices=["off", "add", "relabel"], default="off",
                    help="objeto_pequeno automático según el área relativa de la caja")
    ap.add_argument("--small-area", type=float, default=0.0025, help="fracción del área de imagen (0.25 %%)")
    a = ap.parse_args()

    cfg = yaml.safe_load(rp(a.config).read_text(encoding="utf-8"))
    out = rp(a.out)
    stats = {s: Counter() for s in ("train", "val", "test")}
    n_img, n_skip = Counter(), Counter()

    for src in cfg["sources"]:
        lmap = load_map(src["label_map"])
        name = src["name"]
        print(f"[+] {name} ({src['format']})")
        for img, boxes, (w, h) in READERS[src["format"]](src):
            lines = []
            for b in boxes:
                raw, x1, y1, x2, y2 = b[:5]
                sup = b[5] if len(b) > 5 else ""
                tgt = lmap.get(str(raw).lower().strip(), lmap.get(f"supercat:{str(sup).lower()}"))
                if tgt is None:
                    n_skip[f"{name}:{raw}"] += 1
                    continue
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if (x2 - x1) < a.min_box or (y2 - y1) < a.min_box:
                    continue
                area = (x2 - x1) * (y2 - y1) / (w * h)
                cls_list = [tgt]
                if a.bottle_superclass and tgt in BOTTLES:
                    cls_list.append("botella")
                if a.small_mode != "off" and area < a.small_area:
                    cls_list = ["objeto_pequeno"] if a.small_mode == "relabel" else list(dict.fromkeys(cls_list + ["objeto_pequeno"]))
                for c in cls_list:
                    lines.append((CID[c], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
            if not lines:
                continue
            key = f"{name}_{img.stem}"
            # copias aumentadas de Roboflow (nombre.rf.<hash>) deben caer en el MISMO split que su original
            split = split_of(f"{name}_{img.stem.split('.rf.')[0]}")
            n_img[split] += 1
            for l in lines:
                stats[split][CLASSES[l[0]]] += 1
            if a.dry_run:
                continue
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            dst = out / "images" / split / f"{key}{img.suffix.lower()}"
            if a.symlink:
                dst.unlink(missing_ok=True)
                dst.symlink_to(img.resolve())
            else:
                shutil.copy2(img, dst)
            (out / "labels" / split / f"{key}.txt").write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}" for c, x, y, bw, bh in lines) + "\n"
            )

    print("\n=== Imágenes por split:", dict(n_img))
    for s, c in stats.items():
        print(f"--- {s}: " + ", ".join(f"{k}={c[k]}" for k in CLASSES))
    total = sum((c for c in stats.values()), Counter())
    missing = [k for k in CLASSES if total[k] == 0]
    if missing:
        print(f"\n[AVISO] Clases SIN ejemplos: {missing} -> consigue/etiqueta más datos para ellas.")
    if n_skip:
        print("\nEtiquetas de origen descartadas (sin mapeo):")
        for k, v in n_skip.most_common(15):
            print(f"   {k}: {v}")
    if a.dry_run:
        print("\n(dry-run: no se escribió nada)")


if __name__ == "__main__":
    main()
