import { useEffect, useMemo, useState } from "react";
import ControlPanel from "./components/ControlPanel";
import CountsPanel from "./components/CountsPanel";
import ImageView from "./components/ImageView";
import LiveView from "./components/LiveView";
import { useDetectionSocket } from "./hooks/useDetectionSocket";
import { DEFAULT_CLASSES, DEFAULT_COLORS } from "./config";

const STATUS = {
  open: ["bg-emerald-500", "Backend conectado"],
  connecting: ["bg-amber-400 animate-pulse", "Conectando…"],
  closed: ["bg-rose-500", "Sin conexión (reintentando)"],
};

export default function App() {
  const [tab, setTab] = useState("camera");
  const [settings, setSettings] = useState({
    conf: 0.35, imgsz: 640, sahi: false, sliceSize: 640, overlap: 0.2, disabled: new Set(),
  });
  const [imageResult, setImageResult] = useState(null);
  const socket = useDetectionSocket();

  // Usa la resolución con la que trabaja el modelo cargado (p. ej. 640 si se entrenó a 640).
  const serverImgsz = socket.model?.imgsz;
  useEffect(() => {
    if (serverImgsz) setSettings((s) => ({ ...s, imgsz: serverImgsz }));
  }, [serverImgsz]);

  const classes = socket.model?.classes ?? DEFAULT_CLASSES;
  const colors = { ...DEFAULT_COLORS, ...(socket.model?.colors ?? {}) };

  // Sincroniza parámetros con el servidor (también tras cada reconexión).
  useEffect(() => {
    const enabled = classes.filter((c) => !settings.disabled.has(c));
    socket.sendConfig({
      conf: settings.conf,
      imgsz: settings.imgsz,
      classes: enabled.length === classes.length ? [] : enabled,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.conf, settings.imgsz, settings.disabled, socket.model, socket.sendConfig]);

  const current = tab === "camera" ? socket.result : imageResult;
  // Filtro en cliente (respuesta inmediata al mover el slider o desactivar clases).
  const visible = useMemo(
    () =>
      (current?.detections ?? []).filter(
        (d) => d[5] >= settings.conf && !settings.disabled.has(d[4])
      ),
    [current, settings.conf, settings.disabled]
  );
  const counts = useMemo(() => {
    const c = {};
    visible.forEach((d) => (c[d[4]] = (c[d[4]] || 0) + 1));
    return c;
  }, [visible]);
  const frame = current ? { w: current.width, h: current.height } : null;
  const [dot, label] = STATUS[socket.status];

  return (
    <div className="mx-auto flex min-h-full max-w-[1500px] flex-col gap-4 p-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold">Detector de residuos</h1>
          <p className="text-xs text-slate-400">
            YOLO · {socket.model ? `${socket.model.model.split(/[\\/]/).pop()} · ${socket.model.device} · ${socket.model.imgsz}px` : "…"}
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-300">
          <span className={`h-2.5 w-2.5 rounded-full ${dot}`} />
          {label}
        </div>
      </header>

      {socket.model?.mode === "fallback" && (
        <div className="rounded-lg border border-amber-700 bg-amber-950/40 px-3 py-2 text-sm text-amber-200">
          Modo respaldo: no se encontró <code>weights/best.pt</code>. Se usa el modelo COCO <b>{socket.model.model}</b> con
          las clases traducidas (botella, vidrio, papel, objeto pequeño…) solo para probar el flujo. Entrena tu modelo y reinicia el backend.
        </div>
      )}

      <div className="grid flex-1 gap-4 lg:grid-cols-[1fr_340px]">
        <main className="min-w-0 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <nav className="mb-4 inline-flex rounded-lg bg-slate-800 p-1 text-sm">
            {[["camera", "Cámara en vivo"], ["image", "Imagen local"]].map(([id, text]) => (
              <button key={id} onClick={() => setTab(id)}
                className={`rounded-md px-4 py-1.5 ${tab === id ? "bg-slate-950 font-semibold" : "text-slate-400 hover:text-slate-200"}`}>
                {text}
              </button>
            ))}
          </nav>
          {/* La cámara se libera al salir de su pestaña; la imagen analizada se conserva */}
          <div hidden={tab !== "camera"}>
            {tab === "camera" && <LiveView socket={socket} imgsz={settings.imgsz} visible={visible} frame={frame} />}
          </div>
          <div hidden={tab !== "image"}>
            <ImageView settings={settings} result={imageResult} onResult={setImageResult} visible={tab === "image" ? visible : []} />
          </div>
        </main>

        <aside className="space-y-4">
          <ControlPanel settings={settings} setSettings={setSettings} classes={classes} colors={colors} tab={tab} />
          <CountsPanel classes={classes} colors={colors} counts={counts} stats={socket.stats} tab={tab} status={socket.status} />
        </aside>
      </div>
    </div>
  );
}
