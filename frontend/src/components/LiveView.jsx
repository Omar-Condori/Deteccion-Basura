import { useEffect, useRef, useState } from "react";
import DetectionCanvas from "./DetectionCanvas";
import { useCamera } from "../hooks/useCamera";

/** Vista de cámara en tiempo real: captura -> WebSocket -> canvas. */
export default function LiveView({ socket, imgsz, visible, frame }) {
  const cam = useCamera();
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const capRef = useRef(null);
  const sockRef = useRef(socket);
  sockRef.current = socket;
  const imgszRef = useRef(imgsz);
  imgszRef.current = imgsz;

  // Bucle de envío: un frame nuevo solo cuando el servidor respondió al anterior.
  useEffect(() => {
    if (!cam.active || socket.status !== "open") return;
    let stopped = false;
    let raf;
    const loop = async () => {
      if (stopped) return;
      const v = cam.videoRef.current;
      const s = sockRef.current;
      if (v && v.readyState >= 2 && v.videoWidth && s.canSend()) {
        const k = Math.min(1, imgszRef.current / Math.max(v.videoWidth, v.videoHeight));
        const cap = capRef.current;
        cap.width = Math.round(v.videoWidth * k);
        cap.height = Math.round(v.videoHeight * k);
        cap.getContext("2d").drawImage(v, 0, 0, cap.width, cap.height);
        const blob = await new Promise((r) => cap.toBlob(r, "image/jpeg", 0.8));
        if (blob && !stopped) await s.sendFrame(blob);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [cam.active, socket.status, cam.videoRef]);

  useEffect(() => {
    if (!cam.active) socket.clearResult();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cam.active]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {!cam.active ? (
          <button onClick={() => cam.start()} className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold hover:bg-emerald-500">
            Iniciar cámara
          </button>
        ) : (
          <button onClick={cam.stop} className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold hover:bg-rose-500">
            Detener
          </button>
        )}
        {cam.devices.length > 1 && (
          <select
            value={cam.deviceId}
            onChange={(e) => cam.selectDevice(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-2 text-sm"
          >
            <option value="">Cámara por defecto</option>
            {cam.devices.map((d, i) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label || `Cámara ${i + 1}`}</option>
            ))}
          </select>
        )}
        {cam.error && <span className="text-sm text-rose-400">{cam.error}</span>}
      </div>

      <DetectionCanvas mediaW={dims.w} mediaH={dims.h} detections={cam.active ? visible : []} frameW={frame?.w} frameH={frame?.h}>
        <video
          ref={cam.videoRef}
          muted
          playsInline
          onLoadedMetadata={(e) => setDims({ w: e.target.videoWidth, h: e.target.videoHeight })}
          className="absolute inset-0 h-full w-full object-fill"
        />
        {!cam.active && (
          <div className="absolute inset-0 grid place-items-center text-slate-500">Pulsa «Iniciar cámara» para comenzar</div>
        )}
      </DetectionCanvas>
      <canvas ref={capRef} className="hidden" />
    </div>
  );
}
