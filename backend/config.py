"""Configuración central del backend (variables de entorno con valores por defecto)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Clases objetivo del sistema, en el MISMO orden que ml/data.yaml.
CLASSES: list[str] = [
    "vidrio",
    "carton",
    "papel",
    "metal",
    "botella",
    "botella_vidrio",
    "botella_plastico",
    "objeto_peligroso",
    "objeto_pequeno",
]

# Un color distinto por clase (hex) para las cajas delimitadoras.
CLASS_COLORS: dict[str, str] = {
    "vidrio": "#38BDF8",
    "carton": "#F97316",
    "papel": "#EAB308",
    "metal": "#94A3B8",
    "botella": "#22C55E",
    "botella_vidrio": "#2563EB",
    "botella_plastico": "#A855F7",
    "objeto_peligroso": "#EF4444",
    "objeto_pequeno": "#EC4899",
}

# Modo respaldo: si no existe weights/best.pt se carga un modelo COCO oficial y sus
# clases se traducen a las 9 clases del proyecto SOLO para validar el flujo
# WebSocket + Frontend + Canvas. Las clases COCO sin equivalente se descartan.
FALLBACK_CLASS_MAP: dict[str, str] = {
    "bottle": "botella",
    "wine glass": "vidrio",
    "cup": "vidrio",
    "vase": "vidrio",
    "book": "papel",
    "knife": "objeto_peligroso",
    "scissors": "objeto_peligroso",
    "fork": "metal",
    "spoon": "metal",
    "cell phone": "objeto_pequeno",
    "remote": "objeto_pequeno",
    "mouse": "objeto_pequeno",
    "toothbrush": "objeto_pequeno",
    "clock": "objeto_pequeno",
}


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


def _load_dotenv() -> None:
    """Mini lector de backend/.env (sin dependencias extra)."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    weights_path: Path = field(
        default_factory=lambda: (Path(__file__).resolve().parent / os.getenv("WEIGHTS_PATH", "../weights/best.pt")).resolve()
    )
    fallback_models: list[str] = field(default_factory=lambda: _env_list("FALLBACK_MODELS", "yolo11m.pt,yolov8m.pt"))
    imgsz: int = int(os.getenv("IMGSZ", "1280"))
    conf_default: float = float(os.getenv("CONF_DEFAULT", "0.35"))
    iou: float = float(os.getenv("IOU", "0.5"))
    device: str = os.getenv("DEVICE", "auto")
    half: str = os.getenv("HALF", "auto")
    max_det: int = int(os.getenv("MAX_DET", "300"))
    sahi_slice_size: int = int(os.getenv("SAHI_SLICE_SIZE", "640"))
    sahi_overlap: float = float(os.getenv("SAHI_OVERLAP", "0.2"))
    max_image_side: int = int(os.getenv("MAX_IMAGE_SIDE", "4096"))
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    )


settings = Settings()
