import { useEffect, useRef } from "react";
import { prettyName } from "../config";

function textColorFor(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lum = 0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255);
  return lum > 150 ? "#0f172a" : "#ffffff";
}

/**
 * Contenedor con la relación de aspecto exacta del medio (video/imagen). El <canvas> se superpone
 * al 100 % del área y dibuja las cajas escalando desde las coordenadas del frame enviado al
 * servidor (frameW x frameH) al tamaño real mostrado -> sincronía exacta a cualquier tamaño.
 *
 * props: mediaW/mediaH (dimensiones del medio), detections [[x1,y1,x2,y2,cls,conf,color],...],
 *        frameW/frameH (dimensiones del frame al que se refieren las coordenadas).
 */
export default function DetectionCanvas({ mediaW, mediaH, detections, frameW, frameH, children }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const ratio = mediaW && mediaH ? mediaW / mediaH : 16 / 9;

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const cw = Math.round(wrap.clientWidth * dpr);
      const ch = Math.round(wrap.clientHeight * dpr);
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
      }
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, cw, ch);
      if (!detections?.length || !frameW || !frameH) return;

      const sx = cw / frameW;
      const sy = ch / frameH;
      const fontPx = Math.max(11, Math.round(13 * dpr));
      ctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textBaseline = "middle";
      ctx.lineJoin = "round";

      for (const [x1, y1, x2, y2, cls, conf, color] of detections) {
        const x = x1 * sx, y = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy;
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, 2.5 * dpr);
        ctx.strokeRect(x, y, w, h);

        const label = `${prettyName(cls)} ${Math.round(conf * 100)}%`;
        const padX = 6 * dpr;
        const th = fontPx + 8 * dpr;
        const tw = ctx.measureText(label).width + padX * 2;
        const ly = y - th >= 0 ? y - th : y; // si no cabe arriba, se dibuja dentro de la caja
        const lx = Math.min(Math.max(0, x), Math.max(0, cw - tw));
        ctx.fillStyle = color;
        ctx.fillRect(lx, ly, tw, th);
        ctx.fillStyle = textColorFor(color);
        ctx.fillText(label, lx + padX, ly + th / 2 + 1);
      }
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [detections, frameW, frameH, ratio]);

  return (
    <div
      ref={wrapRef}
      className="relative mx-auto overflow-hidden rounded-xl bg-black"
      style={{ aspectRatio: `${ratio}`, width: `min(100%, ${(75 * ratio).toFixed(2)}vh)` }}
    >
      {children}
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
    </div>
  );
}
