# Detector de residuos en tiempo real (YOLO + FastAPI + React)

Sistema web académico para detectar y clasificar 9 clases: `vidrio`, `carton`, `papel`, `metal`, `botella`,
`botella_vidrio`, `botella_plastico`, `objeto_peligroso`, `objeto_pequeno`.

```
Detector-basura/
├── backend/                 FastAPI + Ultralytics YOLO + SAHI
│   ├── main.py              WebSocket /ws, REST /predict-image, /health, /classes
│   ├── detector.py          carga de modelo (best.pt o respaldo COCO), inferencia normal y SAHI
│   ├── config.py            clases, colores, mapeo COCO->clases (respaldo), variables de entorno
│   ├── requirements.txt
│   └── .env.example
├── ml/
│   ├── data.yaml            definición del dataset (9 clases)
│   ├── train.py             entrenamiento (NO se ejecuta solo)
│   ├── prepare_dataset.py   unifica TACO / ZeroWaste / TrashNet / Roboflow a las 9 clases
│   ├── sources.example.yaml fuentes a unificar
│   └── label_maps/          mapeo de etiquetas por dataset (taco, zerowaste, trashnet, roboflow)
├── frontend/                React + Vite + Tailwind v4
│   └── src/ App.jsx, components/{DetectionCanvas,LiveView,ImageView,ControlPanel,CountsPanel}.jsx,
│            hooks/{useDetectionSocket,useCamera}.js
├── weights/                 aquí va best.pt (tras entrenar)
└── datasets/                aquí va datasets/waste/{images,labels}/{train,val,test}
```

## 1. Puesta en marcha (sin entrenar)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # GPU NVIDIA: instala antes PyTorch CUDA desde pytorch.org
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (otra terminal)
cd frontend
npm install
npm run dev                                              # http://localhost:5173
```

Si `weights/best.pt` **no existe**, el backend descarga `yolo11m.pt` (o `yolov8m.pt`) y traduce clases COCO a las tuyas
(bottle→botella, wine glass/cup/vase→vidrio, book→papel, knife/scissors→objeto_peligroso, fork/spoon→metal,
cell phone/remote/mouse/toothbrush/clock→objeto_pequeno). La interfaz muestra un aviso ámbar. Es solo para probar
el flujo WebSocket → Canvas; la primera ejecución necesita internet para bajar los pesos.
Cuando exista `weights/best.pt` se usa automáticamente (reinicia el backend).

Variables útiles (`backend/.env`): `IMGSZ` (1280), `DEVICE` (auto/cpu/cuda:0/mps), `HALF`, `SAHI_SLICE_SIZE`, `SAHI_OVERLAP`.
Sin GPU, 1280 px es lento: baja la resolución desde el panel (640/960) para el vídeo en vivo.

Para entrenar (desde la raíz del proyecto): `pip install -r ml/requirements.txt`.

## 2. API

| Endpoint | Descripción |
|---|---|
| `WS /ws` | Texto JSON `{"type":"config","conf":0.35,"imgsz":1280,"classes":[...]}`; binario = frame JPEG. Respuesta: `{"type":"result","width","height","inference_ms","detections":[[x1,y1,x2,y2,"clase",conf,"#color"],...],"counts":{...}}`. El cliente espera cada respuesta antes de enviar otro frame. |
| `POST /predict-image` | multipart: `file`, `conf`, `sahi` (bool), `imgsz`, `slice_size`, `overlap`, `classes` (csv). Con `sahi=true` hace inferencia por recortes superpuestos. |
| `GET /health`, `GET /classes` | estado del modelo, modo (`custom`/`fallback`), clases y colores. |

SAHI se aplica a imágenes (REST); en el stream en vivo se usa inferencia estándar por latencia.

## 3. Guía de datasets

### Fuentes
| Dataset | Dónde | Formato | Aporta a |
|---|---|---|---|
| **TACO** | http://tacodataset.org · repo `pdollar/TACO` (`download.py` baja las imágenes + `annotations.json`) | COCO JSON, 60 categorías | vidrio, metal, carton, papel, botellas, objeto_pequeno (tapas, chapas, colillas), objeto_peligroso (baterías, aerosoles) |
| **ZeroWaste** (ZeroWaste-f) | repo `dbash/zerowaste` (enlaces de descarga) | COCO JSON, 4 clases | carton y metal en cinta transportadora |
| **TrashNet** | repo `garythung/trashnet` (`dataset-resized.zip`) | carpetas por clase, **sin cajas** | vidrio, carton, papel, metal (cajas aproximadas; peso bajo) |
| **Roboflow Universe** | https://universe.roboflow.com → buscar: `glass bottle`, `plastic bottle`, `waste detection`, `syringe`, `knife`, `bottle cap`, `cigarette butt` → *Download Dataset → YOLOv8* | YOLO | botellas, objetos peligrosos, pequeños |
| **Open Images V7** (opcional) | `fiftyone.zoo.load_zoo_dataset("open-images-v7", classes=["Knife","Scissors","Bottle"], label_types=["detections"])` | exporta a YOLO | objeto_peligroso, botella |

No verifiqué proyectos concretos de Roboflow: elige los que tengan licencia permisiva, >500 imágenes y etiquetas coherentes.
**Descárgalos sin augmentations** (si no, las copias aumentadas caen en train y val a la vez y inflan las métricas).

### Unificación de etiquetas
1. Descarga cada dataset a `raw/<nombre>/` (carpeta `raw/` en la raíz del proyecto).
2. Copia `ml/sources.example.yaml` → `ml/sources.yaml` y ajusta rutas.
3. Revisa/edita `ml/label_maps/*.yaml` (nombre de origen → clase destino, `null` = descartar). Para Roboflow, usa los nombres exactos del `data.yaml` de cada descarga.
4. Ejecuta `python ml/prepare_dataset.py --config ml/sources.yaml --dry-run`, revisa conteos por clase, y luego sin `--dry-run`.

Opciones: `--bottle-superclass` (cada botella_vidrio/plastico añade también una caja `botella`), `--small-mode add|relabel --small-area 0.0025`
(marca como `objeto_pequeno` las cajas con <0.25 % del área), `--symlink`.

Reglas de diseño recomendadas:
- `botella` genérica solo para botellas cuyo material no se pueda determinar; no la dupliques sobre todas las botellas salvo que lo necesites (cajas solapadas con clases distintas confunden la métrica y la NMS por clase las deja convivir).
- `objeto_pequeno` es una clase de **escala**, no de material: mezcla con claridad en la memoria académica.
- Meta práctica: ≥1 500 instancias por clase y ≥10 % de imágenes de fondo sin objetos. Casi seguro tendrás que **etiquetar tus propias fotos** (Roboflow, CVAT o Label Studio) para `objeto_peligroso`, `botella_*` y vidrio vs. plástico, que las fuentes públicas cubren poco.

### Estructura final (la genera `prepare_dataset.py`)
```
datasets/waste/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt     # clase cx cy w h normalizados
```

## 4. Modelos y entrenamiento

| Variante | Params | Coste (relativo) | Cuándo usarla |
|---|---|---|---|
| `yolo11m.pt` @1280 | ~20 M | 1× | **Recomendada**: mejor equilibrio precisión/velocidad; cabe en GPUs de 12–16 GB con batch 4–8 |
| `yolo11l.pt` @1280 | ~25 M | ~1.3× | +precisión si tienes ≥16–24 GB; batch 4 |
| `yolov8m-p2` @1280 (`--p2`) | ~30 M | ~2–3× | Prioridad en objetos diminutos (cabezal P2, stride 4); más lento. Ultralytics trae `yolov8-p2.yaml` (no existe P2 oficial para YOLO11 en la versión 8.4 revisada) |
| `yolo11s.pt` @960 | ~9 M | ~0.3× | Demos en CPU/portátil |

Cifras de params/coste aproximadas; mídelas en tu hardware con `yolo benchmark`.

### Comando recomendado
```bash
python ml/train.py --smoke                      # 1) verifica el pipeline (1 época a 640)
python ml/train.py                              # 2) YOLO11m, 1280 px, 100 épocas (early stopping con paciencia 20)
python ml/train.py --model yolo11l.pt --batch 4 --name waste_y11l_1280
python ml/train.py --p2 --batch 4               # si objeto_pequeno rinde mal
```
Hiperparámetros por defecto en `train.py`: `imgsz=1280, epochs=100, batch=8, patience=20 (early stopping), optimizer=AdamW, lr0=0.001, lrf=0.01, cos_lr=True,
warmup_epochs=3, weight_decay=0.0005, close_mosaic=10, mosaic=1.0, mixup=0.1, copy_paste=0.1, scale=0.5, translate=0.1, degrees=10,
fliplr=0.5, flipud=0.1, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, erasing=0.2, box=7.5, cls=0.5, dfl=1.5, cache=ram`.
Notas: sube `hsv_s/hsv_v` solo si el color no es discriminante (vidrio vs. plástico depende de brillo/transparencia, no abuses);
si hay desbalance fuerte sube `cls` (0.7–1.0) y añade imágenes de las clases raras; para <2 000 imágenes sube a `--epochs 200`.

Al terminar, `best.pt` se copia a `weights/best.pt` y el backend lo usa al reiniciar. Evalúa con `mAP50-95` por clase y la matriz de confusión
(en `runs/detect/<name>/`); revisa especialmente botella vs. botella_vidrio vs. botella_plastico.
