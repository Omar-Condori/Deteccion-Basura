import { useCallback, useEffect, useRef, useState } from "react";
import DetectionCanvas from "./DetectionCanvas";
import { API_URL } from "../config";

/** Subida de imágenes locales + análisis por REST (/predict-image), con SAHI opcional. */
export default function ImageView({ settings, result, onResult, visible, onBusy }) {
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState(null);
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => () => url && URL.revokeObjectURL(url), [url]);

  const analyze = useCallback(
    async (f = file) => {
      if (!f) return;
      setLoading(true);
      setError(null);
      onBusy?.(true);
      try {
        const fd = new FormData();
        fd.append("file", f);
        // Se pide con umbral bajo y el slider filtra en el cliente: al bajarlo aparecen más cajas sin re-analizar.
        fd.append("conf", "0.05");
        fd.append("sahi", String(settings.sahi));
        fd.append("imgsz", String(settings.imgsz));
        fd.append("slice_size", String(settings.sliceSize));
        fd.append("overlap", String(settings.overlap));
        const res = await fetch(`${API_URL}/predict-image`, { method: "POST", body: fd });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
        onResult(await res.json());
      } catch (e) {
        setError(e.message === "Failed to fetch" ? "No se pudo conectar con el backend" : e.message);
      } finally {
        setLoading(false);
        onBusy?.(false);
      }
    },
    [file, settings.sahi, settings.imgsz, settings.sliceSize, settings.overlap, onResult, onBusy]
  );

  const pick = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setUrl(URL.createObjectURL(f));
    onResult(null);
    analyze(f);
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files?.[0])} />
        <button onClick={() => inputRef.current.click()} className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold hover:bg-sky-500">
          Subir imagen
        </button>
        <button
          disabled={!file || loading}
          onClick={() => analyze()}
          className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold hover:bg-emerald-500 disabled:opacity-40"
        >
          {loading ? "Analizando…" : settings.sahi ? "Re-analizar con SAHI" : "Re-analizar"}
        </button>
        {result && (
          <span className="text-xs text-slate-400">
            {result.sahi ? "SAHI · " : ""}{result.inference_ms} ms · {result.width}×{result.height}px
          </span>
        )}
        {error && <span className="text-sm text-rose-400">{error}</span>}
      </div>

      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          pick(e.dataTransfer.files?.[0]);
        }}
      >
        {url ? (
          <DetectionCanvas mediaW={dims.w} mediaH={dims.h} detections={visible} frameW={result?.width} frameH={result?.height}>
            <img
              src={url}
              alt="Imagen a analizar"
              onLoad={(e) => setDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
              className="absolute inset-0 h-full w-full object-fill"
            />
            {loading && <div className="absolute inset-0 grid place-items-center bg-black/50 text-sm">Analizando…</div>}
          </DetectionCanvas>
        ) : (
          <div
            onClick={() => inputRef.current.click()}
            className="grid h-72 cursor-pointer place-items-center rounded-xl border-2 border-dashed border-slate-700 text-slate-500 hover:border-slate-500"
          >
            Arrastra una imagen aquí o haz clic para seleccionarla
          </div>
        )}
      </div>
    </div>
  );
}
