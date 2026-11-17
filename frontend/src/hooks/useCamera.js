import { useCallback, useEffect, useRef, useState } from "react";

/** Captura de cámara con getUserMedia. */
export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState("");

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
  }, []);

  const start = useCallback(
    async (id = deviceId) => {
      stop();
      setError(null);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            deviceId: id ? { exact: id } : undefined,
            facingMode: id ? undefined : { ideal: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
        });
        streamRef.current = stream;
        const v = videoRef.current;
        v.srcObject = stream;
        await v.play();
        setActive(true);
        const all = await navigator.mediaDevices.enumerateDevices();
        setDevices(all.filter((d) => d.kind === "videoinput"));
      } catch (e) {
        setError(e.name === "NotAllowedError" ? "Permiso de cámara denegado" : e.message);
      }
    },
    [deviceId, stop]
  );

  const selectDevice = useCallback((id) => {
    setDeviceId(id);
    start(id);
  }, [start]);

  useEffect(() => stop, [stop]);

  return { videoRef, active, error, devices, deviceId, start, stop, selectDevice };
}
