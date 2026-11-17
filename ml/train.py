"""Entrenamiento de YOLO para detección de residuos (9 clases).

NO se ejecuta automáticamente: úsalo cuando tengas el dataset en datasets/waste.

Ejemplos (desde la raíz del proyecto):
    python ml/train.py                                   # YOLO11m @1280 (recomendado)
    python ml/train.py --model yolo11l.pt --batch 4      # más precisión, más VRAM
    python ml/train.py --p2                              # YOLOv8m-P2: cabezal extra para objetos diminutos
    python ml/train.py --smoke                           # prueba rápida de 1 época (verifica el pipeline)
    python ml/train.py --resume                          # reanudar el último entrenamiento
Al terminar copia runs/detect/<name>/weights/best.pt -> weights/best.pt (lo que carga el backend).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "ml" / "data.yaml"
RESOLVED_YAML = ROOT / "ml" / ".data.resolved.yaml"


def resolve_data_yaml() -> Path:
    """Escribe una copia de data.yaml con `path` absoluto y valida que el dataset exista."""
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    ds = (DATA_YAML.parent / cfg["path"]).resolve()
    for split in ("train", "val"):
        if not (ds / cfg[split]).exists():
            raise SystemExit(
                f"[ERROR] No existe {ds / cfg[split]}.\n"
                "Prepara el dataset primero (ver README.md y ml/prepare_dataset.py)."
            )
    cfg["path"] = str(ds)
    RESOLVED_YAML.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return RESOLVED_YAML


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11m.pt", help="pesos .pt preentrenados o arquitectura .yaml")
    p.add_argument("--p2", action="store_true",
                   help="usa yolov8m-p2.yaml (cabezal P2/stride 4) cargando pesos yolov8m.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=float, default=8, help="entero, o -1 = auto (60%% VRAM), o 0.7 = 70%% VRAM")
    p.add_argument("--patience", type=int, default=20, help="early stopping: corta si no mejora en N épocas")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None, help="0, 0,1, cpu, mps (por defecto: automático)")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"])
    p.add_argument("--name", default=None, help="por defecto: waste_<modelo>_<imgsz>")
    p.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 época, imgsz 640, 50 %% del dataset (solo verificar)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    from ultralytics import YOLO

    data = resolve_data_yaml()

    if a.resume:
        if a.name:
            last = Path(a.project) / a.name / "weights" / "last.pt"
        else:  # el entrenamiento más reciente
            cands = sorted(Path(a.project).glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
            if not cands:
                raise SystemExit("[ERROR] No hay entrenamientos previos que reanudar.")
            last = cands[-1]
        print(f"Reanudando desde {last}")
        YOLO(str(last)).train(resume=True)
        return

    if not a.name:
        a.name = "smoke" if a.smoke else (f"waste_v8m_p2_{a.imgsz}" if a.p2 else f"waste_{Path(a.model).stem}_{a.imgsz}")

    if a.p2:
        model = YOLO("yolov8m-p2.yaml")
        model.load("yolov8m.pt")  # transfiere los pesos COCO compatibles
    else:
        model = YOLO(a.model)

    hyp = dict(
        # --- optimización ---
        optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0, close_mosaic=10,
        # --- aumentación (materiales: vidrio/plástico dependen del color y brillo) ---
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0,
        fliplr=0.5, flipud=0.1,
        mosaic=1.0, mixup=0.1, copy_paste=0.1, erasing=0.2,
        # --- pérdidas ---
        box=7.5, cls=0.5, dfl=1.5,
        # --- NMS en validación ---
        iou=0.6, max_det=300,
    )
    if a.smoke:
        a.epochs, a.imgsz, a.batch, a.cache = 1, 640, 4, "false"
        hyp["fraction"] = 0.5

    results = model.train(
        data=str(data),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=int(a.batch) if a.batch >= 1 else a.batch,
        patience=a.patience,
        workers=a.workers,
        device=a.device,
        cache=False if a.cache == "false" else a.cache,
        project=a.project,
        name=a.name,
        exist_ok=False,
        pretrained=True,
        amp=True,
        plots=True,
        seed=0,
        deterministic=True,
        **hyp,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists() and not a.smoke:
        dst = ROOT / "weights" / "best.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(best, dst)
        print(f"\n[OK] Pesos copiados a {dst}. Reinicia el backend para usarlos.")
    metrics = YOLO(str(best)).val(data=str(data), imgsz=a.imgsz, split="test" if (Path(yaml.safe_load(data.read_text())["path"]) / "images" / "test").exists() else "val")
    print("mAP50-95:", metrics.box.map, "| mAP50:", metrics.box.map50)


if __name__ == "__main__":
    main()
