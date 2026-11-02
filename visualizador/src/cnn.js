// Pasada hacia adelante (forward) de la CNN entrenada en Keras, escrita a mano.
// Sirve para mostrar cada paso en la visualización: convolución, ReLU, pooling, capa densa y softmax.

export function decodificar(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

export function construirModelo(pesos) {
  const capa = (p) => ({ w: decodificar(p.w), b: decodificar(p.b), shape: p.w_shape });
  return { c1: capa(pesos.conv1), c2: capa(pesos.conv2), d1: capa(pesos.dense1), d2: capa(pesos.dense2) };
}

// Convolución 3x3 con padding "same" + ReLU. Datos en orden (alto, ancho, canal), igual que Keras.
function conv(x, H, W, cin, k, cout) {
  const out = new Float32Array(H * W * cout);
  const { w, b } = k;
  for (let y = 0; y < H; y++) {
    for (let xx = 0; xx < W; xx++) {
      const o = (y * W + xx) * cout;
      for (let co = 0; co < cout; co++) out[o + co] = b[co];
      for (let ky = 0; ky < 3; ky++) {
        const yy = y + ky - 1;
        if (yy < 0 || yy >= H) continue;
        for (let kx = 0; kx < 3; kx++) {
          const x2 = xx + kx - 1;
          if (x2 < 0 || x2 >= W) continue;
          const ib = (yy * W + x2) * cin;
          const wb = (ky * 3 + kx) * cin * cout;
          for (let ci = 0; ci < cin; ci++) {
            const v = x[ib + ci];
            if (v === 0) continue;
            const wo = wb + ci * cout;
            for (let co = 0; co < cout; co++) out[o + co] += v * w[wo + co];
          }
        }
      }
      for (let co = 0; co < cout; co++) if (out[o + co] < 0) out[o + co] = 0;
    }
  }
  return out;
}

// Max pooling 2x2: se queda con el valor más grande de cada zona.
function pool(x, H, W, C) {
  const h = H >> 1, w = W >> 1;
  const out = new Float32Array(h * w * C);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < w; xx++) {
      for (let c = 0; c < C; c++) {
        const a = x[((2 * y) * W + 2 * xx) * C + c];
        const b = x[((2 * y) * W + 2 * xx + 1) * C + c];
        const d = x[((2 * y + 1) * W + 2 * xx) * C + c];
        const e = x[((2 * y + 1) * W + 2 * xx + 1) * C + c];
        out[(y * w + xx) * C + c] = Math.max(a, b, d, e);
      }
    }
  }
  return out;
}

function densa(x, k, relu) {
  const [nin, nout] = k.shape;
  const out = new Float32Array(nout);
  for (let o = 0; o < nout; o++) out[o] = k.b[o];
  for (let i = 0; i < nin; i++) {
    const v = x[i];
    if (v === 0) continue;
    const base = i * nout;
    for (let o = 0; o < nout; o++) out[o] += v * k.w[base + o];
  }
  if (relu) for (let o = 0; o < nout; o++) if (out[o] < 0) out[o] = 0;
  return out;
}

export function adelante(m, img) {
  const a1 = conv(img, 28, 28, 1, m.c1, 32);
  const p1 = pool(a1, 28, 28, 32);
  const a2 = conv(p1, 14, 14, 32, m.c2, 64);
  const p2 = pool(a2, 14, 14, 64);
  const d1 = densa(p2, m.d1, true);
  const z = densa(d1, m.d2, false);
  let mx = -Infinity;
  for (const v of z) mx = Math.max(mx, v);
  const e = Array.from(z, (v) => Math.exp(v - mx));
  const suma = e.reduce((s, v) => s + v, 0);
  const probs = e.map((v) => v / suma);
  return { a1, p1, a2, p2, d1, z, probs };
}
