export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const WS_URL = API_URL.replace(/^http/, "ws") + "/ws";

// Clases y colores por defecto (el backend los sobrescribe al conectar con el mensaje "ready").
export const DEFAULT_CLASSES = [
  "vidrio", "carton", "papel", "metal", "botella",
  "botella_vidrio", "botella_plastico", "objeto_peligroso", "objeto_pequeno",
];
export const DEFAULT_COLORS = {
  vidrio: "#38BDF8", carton: "#F97316", papel: "#EAB308", metal: "#94A3B8", botella: "#22C55E",
  botella_vidrio: "#2563EB", botella_plastico: "#A855F7", objeto_peligroso: "#EF4444", objeto_pequeno: "#EC4899",
};

export const prettyName = (c) => c.replaceAll("_", " ");
