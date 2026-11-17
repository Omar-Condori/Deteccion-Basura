"""API FastAPI: WebSocket de tiempo real + REST /predict-image (con SAHI opcional).

Ejecutar (desde la carpeta backend/):
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import CLASS_COLORS, CLASSES, settings
from detector import Detector, color_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

detector = Detector(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    info = await asyncio.to_thread(detector.load)
    log.info("Listo: modo=%s modelo=%s dispositivo=%s", info.mode, info.name, info.device)
    yield


app = FastAPI(title="Detector de residuos - Visión por computador", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- utilidades
def _model_payload() -> dict:
    i = detector.info
    classes = i.classes
    return {
        "mode": i.mode,
        "model": i.name,
        "device": i.device,
        "half": i.half,
        "imgsz": settings.imgsz,
        "classes": classes,
        "colors": {c: color_for(c) for c in classes},
        "sahi_slice_size": settings.sahi_slice_size,
        "sahi_overlap": settings.sahi_overlap,
    }


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # respeta la orientación EXIF
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    h, w = img.shape[:2]
    m = settings.max_image_side
    if max(h, w) > m:
        s = m / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _allowed(classes) -> set[str] | None:
    if not classes:
        return None
    if isinstance(classes, str):
        classes = [c for c in classes.split(",") if c.strip()]
    return {c.strip() for c in classes} or None


def _result(img: np.ndarray, dets: list, ms: float, **extra) -> dict:
    h, w = img.shape[:2]
    return {
        "type": "result",
        "width": w,
        "height": h,
        "inference_ms": ms,
        "mode": detector.info.mode,
        "detections": dets,  # [x1, y1, x2, y2, clase, confianza, color]
        "counts": detector.counts(dets),
        **extra,
    }


# ----------------------------------------------------------------------- REST
@app.get("/health")
def health():
    return {"status": "ok", **_model_payload()}


@app.get("/classes")
def classes():
    return {"classes": CLASSES, "colors": CLASS_COLORS}


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
    sahi: bool = Form(False),
    imgsz: int = Form(settings.imgsz),
    slice_size: int = Form(settings.sahi_slice_size),
    overlap: float = Form(settings.sahi_overlap),
    classes: str = Form(""),
):
    """Analiza una imagen subida. Con sahi=true usa inferencia por recortes (objetos pequeños)."""
    if not (0.0 < conf < 1.0):
        raise HTTPException(400, "conf debe estar entre 0 y 1")
    try:
        img = _decode(await file.read())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    allowed = _allowed(classes)
    if sahi:
        fn = lambda: detector.timed(  # noqa: E731
            detector.predict_sahi, img, conf=conf, slice_size=slice_size, overlap=overlap, allowed=allowed
        )
    else:
        fn = lambda: detector.timed(detector.predict, img, conf=conf, imgsz=imgsz, allowed=allowed)  # noqa: E731
    dets, ms = await asyncio.to_thread(fn)
    return _result(img, dets, ms, sahi=sahi)


# ------------------------------------------------------------------ WebSocket
@app.websocket("/ws")
async def ws_detect(ws: WebSocket):
    """Protocolo:
    servidor -> cliente  {"type":"ready", ...modelo/clases/colores}
    cliente  -> servidor texto JSON {"type":"config","conf":0.35,"imgsz":1280,"classes":["vidrio",...]}
    cliente  -> servidor bytes (frame JPEG/PNG/WebP)
    servidor -> cliente  {"type":"result","width":W,"height":H,"detections":[[x1,y1,x2,y2,clase,conf,color],...],...}
    El cliente debe esperar el resultado antes de enviar el siguiente frame (control de latencia).
    """
    await ws.accept()
    cfg = {"conf": settings.conf_default, "imgsz": settings.imgsz, "iou": settings.iou, "classes": None}
    await ws.send_json({"type": "ready", **_model_payload()})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "config":
                    if "conf" in data:
                        cfg["conf"] = min(max(float(data["conf"]), 0.01), 0.99)
                    if "imgsz" in data:
                        cfg["imgsz"] = int(min(max(int(data["imgsz"]), 320), 1920))
                    if "iou" in data:
                        cfg["iou"] = min(max(float(data["iou"]), 0.1), 0.95)
                    if "classes" in data:
                        cfg["classes"] = _allowed(data["classes"])
                    await ws.send_json({"type": "config_ack", **{k: (sorted(v) if isinstance(v, set) else v) for k, v in cfg.items()}})
                continue
            frame = msg.get("bytes")
            if not frame:
                continue
            try:
                img = _decode(frame)
            except ValueError:
                await ws.send_json({"type": "error", "message": "frame inválido"})
                continue
            dets, ms = await asyncio.to_thread(
                lambda: detector.timed(
                    detector.predict, img, conf=cfg["conf"], imgsz=cfg["imgsz"], iou=cfg["iou"], allowed=cfg["classes"]
                )
            )
            await ws.send_json(_result(img, dets, ms))
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("Error en WebSocket")
        await ws.close(code=1011)
