import { prettyName } from "../config";

export default function CountsPanel({ classes, colors, counts, stats, tab, status }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const max = Math.max(1, ...Object.values(counts));
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Conteo en vivo</h2>
        <span className="text-2xl font-bold tabular-nums">{total}</span>
      </div>
      <ul className="space-y-2">
        {classes.map((c) => {
          const n = counts[c] || 0;
          return (
            <li key={c} className={n ? "" : "opacity-40"}>
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-sm" style={{ background: colors[c] }} />
                  {prettyName(c)}
                </span>
                <span className="font-mono tabular-nums">{n}</span>
              </div>
              <div className="mt-1 h-1 rounded bg-slate-800">
                <div className="h-1 rounded transition-all" style={{ width: `${(n / max) * 100}%`, background: colors[c] }} />
              </div>
            </li>
          );
        })}
      </ul>
      {tab === "camera" && (
        <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-slate-800 pt-3 text-center text-xs text-slate-400">
          <div><dt>FPS</dt><dd className="font-mono text-base text-slate-100">{status === "open" ? stats.fps : "–"}</dd></div>
          <div><dt>Inferencia</dt><dd className="font-mono text-base text-slate-100">{stats.inferenceMs || "–"}<span className="text-xs"> ms</span></dd></div>
          <div><dt>Latencia</dt><dd className="font-mono text-base text-slate-100">{stats.rtt || "–"}<span className="text-xs"> ms</span></dd></div>
        </dl>
      )}
    </section>
  );
}
