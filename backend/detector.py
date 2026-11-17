"""Motor de inferencia: Ultralytics YOLO (+ SAHI opcional) con modelo de respaldo."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from config import CLASS_COLORS, FALLBACK_CLASS_MAP, Settings

log = logging.getLogger("detector")

# Formato público de una detección: [x1, y1, x2, y2, clase, confianza, color]
Detection = list


def color_for(name: str) -> str:
    """Color de la clase; para clases desconocidas genera uno estable a partir del nombre."""
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    h = hashlib.md5(name.encode()).digest()
    return "#{:02X}{:02X}{:02X}".format(*(80 + h[i] % 150 for i in range(3)))


@dataclass
class ModelInfo:
    mode: str          # "custom" (weights/best.pt) | "fallback" (modelo COCO oficial)
    name: str          # ruta o nombre del archivo de pesos cargado
    device: str
    classes: list[str]  # clases que expone el sistema en este modo
    half: bool


class Detector:
    def __init__(self, settings: Settings):
        self.s = settings
        self.model = None
        self.info: ModelInfo | None = None
        self._sahi_model = None
        self._device = "cpu"
        self._half = False
        # Un solo hilo de inferencia a la vez (el modelo no es thread-safe).
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ carga
    def _pick_device(self) -> str:
        if self.s.device != "auto":
            return self.s.device
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> ModelInfo:
        from ultralytics import YOLO

        self._device = self._pick_device()
        self._half = self._device.startswith("cuda") and self.s.half != "false"
        if self.s.half == "true":
            self._half = self._device.startswith("cuda")

        if self.s.weights_path.exists():
            self.model = YOLO(str(self.s.weights_path))
            mode, name = "custom", str(self.s.weights_path)
            log.info("Modelo personalizado cargado: %s", name)
        else:
            log.warning("No existe %s -> usando modelo de respaldo (COCO).", self.s.weights_path)
            last_err: Exception | None = None
            for candidate in self.s.fallback_models:
                try:
                    self.model = YOLO(candidate)  # se descarga automáticamente la 1ª vez
                    mode, name = "fallback", candidate
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    log.error("No se pudo cargar %s: %s", candidate, exc)
            else:
                raise RuntimeError(f"No se pudo cargar ningún modelo de respaldo: {last_err}")

        names = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
        exposed = names if mode == "custom" else sorted(set(FALLBACK_CLASS_MAP.values()))
        self.info = ModelInfo(mode=mode, name=name, device=self._device, classes=exposed, half=self._half)
        self._warmup()
        return self.info

    def _warmup(self) -> None:
        dummy = np.zeros((self.s.imgsz, self.s.imgsz, 3), dtype=np.uint8)
        try:
            self.predict(dummy, conf=0.5)
        except Exception as exc:  # noqa: BLE001
            log.warning("Warm-up falló (no es crítico): %s", exc)

    # ------------------------------------------------------------ utilidades
    def _map_name(self, raw: str) -> str | None:
        """Traduce el nombre devuelto por el modelo al nombre de clase del sistema."""
        if self.info and self.info.mode == "fallback":
            return FALLBACK_CLASS_MAP.get(raw)
        return raw

    def _pack(self, box: Iterable[float], raw_name: str, conf: float, allowed: set[str] | None) -> Detection | None:
        name = self._map_name(raw_name)
        if name is None or (allowed is not None and name not in allowed):
            return None
        x1, y1, x2, y2 = (round(float(v), 1) for v in box)
        return [x1, y1, x2, y2, name, round(float(conf), 4), color_for(name)]

    # ------------------------------------------------------------- inferencia
    def predict(
        self,
        img_bgr: np.ndarray,
        conf: float | None = None,
        imgsz: int | None = None,
        iou: float | None = None,
        allowed: set[str] | None = None,
    ) -> list[Detection]:
        """Inferencia estándar sobre un frame BGR (tiempo real)."""
        with self._lock:
            results = self.model.predict(
                source=img_bgr,
                imgsz=imgsz or self.s.imgsz,
                conf=conf if conf is not None else self.s.conf_default,
                iou=iou if iou is not None else self.s.iou,
                device=self._device,
                half=self._half,
                max_det=self.s.max_det,
                verbose=False,
            )
        r = results[0]
        out: list[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, confs, clss):
            det = self._pack(box, r.names[int(k)], c, allowed)
            if det:
                out.append(det)
        return out

    def _get_sahi(self):
        if self._sahi_model is None:
            from sahi import AutoDetectionModel

            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model=self.model,  # reutiliza el modelo ya cargado (sin duplicar memoria)
                confidence_threshold=self.s.conf_default,
                device=self._device,
                image_size=self.s.sahi_slice_size,
            )
        return self._sahi_model

    def predict_sahi(
        self,
        img_bgr: np.ndarray,
        conf: float | None = None,
        slice_size: int | None = None,
        overlap: float | None = None,
        allowed: set[str] | None = None,
    ) -> list[Detection]:
        """Inferencia por recortes superpuestos (SAHI) para objetos diminutos."""
        from sahi.predict import get_sliced_prediction

        slice_size = slice_size or self.s.sahi_slice_size
        overlap = self.s.sahi_overlap if overlap is None else overlap
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # SAHI espera RGB
        with self._lock:
            sm = self._get_sahi()
            sm.confidence_threshold = conf if conf is not None else self.s.conf_default
            sm.image_size = slice_size
            result = get_sliced_prediction(
                rgb,
                sm,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                perform_standard_pred=True,   # añade la pasada global para objetos grandes
                postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5,
                verbose=0,
            )
        out: list[Detection] = []
        for op in result.object_prediction_list:
            det = self._pack(op.bbox.to_xyxy(), op.category.name, op.score.value, allowed)
            if det:
                out.append(det)
        return out

    # ------------------------------------------------------------ helpers API
    @staticmethod
    def counts(dets: list[Detection]) -> dict[str, int]:
        c: dict[str, int] = {}
        for d in dets:
            c[d[4]] = c.get(d[4], 0) + 1
        return c

    def timed(self, fn, *args, **kwargs) -> tuple[list[Detection], float]:
        t0 = time.perf_counter()
        dets = fn(*args, **kwargs)
        return dets, round((time.perf_counter() - t0) * 1000, 1)
