import { prettyName } from "../config";

const Row = ({ label, value, children }) => (
  <div>
    <div className="mb-1 flex justify-between text-xs text-slate-400">
      <span>{label}</span>
      <span className="font-mono text-slate-200">{value}</span>
    </div>
    {children}
  </div>
);

export default function ControlPanel({ settings, setSettings, classes, colors, tab }) {
  const set = (patch) => setSettings((s) => ({ ...s, ...patch }));
  const toggle = (c) =>
    setSettings((s) => {
      const d = new Set(s.disabled);
      d.has(c) ? d.delete(c) : d.add(c);
      return { ...s, disabled: d };
    });

  return (
    <section className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Controles</h2>

      <Row label="Umbral de confianza" value={`${Math.round(settings.conf * 100)}%`}>
        <input type="range" min="5" max="95" step="1" value={Math.round(settings.conf * 100)}
          onChange={(e) => set({ conf: Number(e.target.value) / 100 })} className="w-full" />
      </Row>

      <Row label="Resolución de entrada (imgsz)" value={settings.imgsz}>
        <select value={settings.imgsz} onChange={(e) => set({ imgsz: Number(e.target.value) })}
          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm">
          {[640, 960, 1280].map((v) => <option key={v} value={v}>{v} × {v}</option>)}
        </select>
      </Row>

      <div className={`space-y-3 rounded-lg border p-3 ${settings.sahi ? "border-emerald-700 bg-emerald-950/30" : "border-slate-800"}`}>
        <label className="flex cursor-pointer items-center justify-between gap-2 text-sm">
          <span>
            Objetos pequeños (SAHI)
            <span className="block text-xs text-slate-500">
              {tab === "image" ? "Inferencia por recortes superpuestos" : "Solo disponible al analizar imágenes"}
            </span>
          </span>
          <input type="checkbox" checked={settings.sahi} onChange={(e) => set({ sahi: e.target.checked })} className="h-4 w-4 accent-emerald-500" />
        </label>
        {settings.sahi && (
          <>
            <Row label="Tamaño de recorte" value={`${settings.sliceSize}px`}>
              <input type="range" min="256" max="1280" step="64" value={settings.sliceSize}
                onChange={(e) => set({ sliceSize: Number(e.target.value) })} className="w-full" />
            </Row>
            <Row label="Solape" value={`${Math.round(settings.overlap * 100)}%`}>
              <input type="range" min="5" max="50" step="5" value={Math.round(settings.overlap * 100)}
                onChange={(e) => set({ overlap: Number(e.target.value) / 100 })} className="w-full" />
            </Row>
          </>
        )}
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
          <span>Clases a mostrar</span>
          <span className="space-x-2">
            <button className="hover:text-slate-200" onClick={() => set({ disabled: new Set() })}>todas</button>
            <button className="hover:text-slate-200" onClick={() => set({ disabled: new Set(classes) })}>ninguna</button>
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {classes.map((c) => {
            const on = !settings.disabled.has(c);
            return (
              <button key={c} onClick={() => toggle(c)}
                className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition ${on ? "border-slate-600 bg-slate-800" : "border-slate-800 text-slate-600 line-through"}`}>
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: on ? colors[c] : "#475569" }} />
                {prettyName(c)}
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
