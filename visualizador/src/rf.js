// Random Forest de scikit-learn, recorrido a mano. Cada arbol es una lista de nodos en preorden:
// el hijo izquierdo es siempre el nodo siguiente y el derecho esta a "der" lugares de distancia.

export const HOJA = 65535;

export function b64ABytes(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

// "crudo" son los bytes ya descomprimidos: feat (uint16) + thr (uint8) + der (uint16), todos de N nodos
export function construirBosque(crudo, meta) {
  const N = meta.nodos;
  const copia = crudo.slice(); // buffer propio y alineado
  return {
    arboles: meta.arboles,
    inicio: meta.inicio,
    feat: new Uint16Array(copia.buffer, 0, N),
    thr: new Uint8Array(copia.buffer, 2 * N, N),
    der: new Uint16Array(copia.buffer.slice(3 * N, 5 * N)),
  };
}

// Devuelve la clase que vota cada arbol para la imagen "pix" (enteros 0-255, 784 valores)
export function votarArbol(b, t, pix) {
  let i = b.inicio[t];
  for (;;) {
    const f = b.feat[i];
    if (f === HOJA) return b.thr[i];
    i = pix[f] <= b.thr[i] ? i + 1 : i + b.der[i];
  }
}

export function votar(b, pix) {
  const porArbol = new Uint8Array(b.arboles);
  const votos = new Int32Array(10);
  for (let t = 0; t < b.arboles; t++) { porArbol[t] = votarArbol(b, t, pix); votos[porArbol[t]]++; }
  return { porArbol, votos, probs: Array.from(votos, (v) => v / b.arboles) };
}

// Las preguntas que hace un arbol hasta llegar a su hoja
export function camino(b, t, pix) {
  const pasos = [];
  let i = b.inicio[t];
  for (;;) {
    const f = b.feat[i];
    if (f === HOJA) return { pasos, clase: b.thr[i] };
    const si = pix[f] <= b.thr[i];
    pasos.push({ pixel: f, umbral: b.thr[i], valor: pix[f], izq: si });
    i = si ? i + 1 : i + b.der[i];
  }
}
