import { useCallback, useEffect, useRef, useState } from "react";
import { WS_URL } from "../config";

/**
 * Conexión WebSocket con control de flujo: solo se envía un frame cuando el anterior
 * ya fue respondido (mínima latencia, sin cola creciente). Reconecta automáticamente.
 */
export function useDetectionSocket() {
  const wsRef = useRef(null);
  const inflight = useRef(false);
  const lastSent = useRef(0);
  const cfgRef = useRef(null);
  const [status, setStatus] = useState("connecting"); // connecting | open | closed
  const [model, setModel] = useState(null); // mensaje "ready" del servidor
  const [result, setResult] = useState(null); // último resultado {width,height,detections,counts,...}
  const [stats, setStats] = useState({ fps: 0, rtt: 0, inferenceMs: 0 });
  const fpsRef = useRef({ n: 0, t0: performance.now() });

  useEffect(() => {
    let alive = true;
    let timer;
    const connect = () => {
      setStatus("connecting");
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => {
        setStatus("open");
        inflight.current = false;
        if (cfgRef.current) ws.send(JSON.stringify({ type: "config", ...cfgRef.current }));
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        const msg = JSON.parse(ev.data);
        if (msg.type === "ready") setModel(msg);
        else if (msg.type === "result") {
          inflight.current = false;
          const now = performance.now();
          const f = fpsRef.current;
          f.n += 1;
          const rtt = now - lastSent.current;
          if (now - f.t0 >= 1000) {
            setStats({ fps: Math.round((f.n * 1000) / (now - f.t0)), rtt: Math.round(rtt), inferenceMs: msg.inference_ms });
            f.n = 0;
            f.t0 = now;
          }
          setResult(msg);
        } else if (msg.type === "error") inflight.current = false;
      };
      ws.onclose = () => {
        setStatus("closed");
        inflight.current = false;
        if (alive) timer = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      alive = false;
      clearTimeout(timer);
      wsRef.current?.close();
    };
  }, []);

  /** Envía parámetros (conf, imgsz, classes[]). Se re-envían tras reconectar. */
  const sendConfig = useCallback((cfg) => {
    cfgRef.current = cfg;
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "config", ...cfg }));
  }, []);

  /** Envía un frame (Blob/ArrayBuffer). Devuelve false si el anterior sigue en proceso. */
  const sendFrame = useCallback(async (blob) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || inflight.current) return false;
    inflight.current = true;
    lastSent.current = performance.now();
    ws.send(await blob.arrayBuffer());
    return true;
  }, []);

  /** true si hay conexión abierta y ningún frame pendiente (evita capturar frames inútiles). */
  const canSend = useCallback(
    () => wsRef.current?.readyState === WebSocket.OPEN && !inflight.current,
    []
  );

  const clearResult = useCallback(() => setResult(null), []);

  return { status, model, result, stats, sendFrame, canSend, sendConfig, clearResult };
}
