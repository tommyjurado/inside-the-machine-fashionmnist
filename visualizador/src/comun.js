// Piezas compartidas por las dos escenas: paleta, formato de números, rampas de color,
// textos en 3D y grillas de mapas de activación (un atlas por capa = un solo draw call).
import * as THREE from 'three';

export const T = {
  fondo: '#090D12', superficie: '#111923', panel: '#17212B', borde: '#293641',
  texto: '#F3F6F8', tenue: '#91A0AC', cnn: '#43D9E8', rf: '#F0A45A', ok: '#6FCF97', mal: '#E5736A',
};
export const CLASES = ['Remera', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota'];
export const FUENTE_MONO = '"JetBrains Mono Variable", "JetBrains Mono", Consolas, monospace';
export const FUENTE_SANS = '"Inter Variable", Inter, "Segoe UI", system-ui, sans-serif';

export const lim = (v) => Math.min(1, Math.max(0, v));
export const suave = (v) => { v = lim(v); return v * v * (3 - 2 * v); };
export const pc = (p, d = 1) => (p * 100).toFixed(d).replace('.', ',');
export const num = (v, d = 2) => v.toFixed(d).replace('.', ',').replace('-', '−');
export const miles = (n) => n.toLocaleString('es-AR');

// Progreso de cada paso de una secuencia en el tiempo t (ms): devuelve {id: 0..1}
export function progresos(pasos, t) {
  const u = {};
  let ini = 0;
  for (const p of pasos) { u[p.id] = lim((t - ini) / p.dur); ini += p.dur; }
  return u;
}
export const duracion = (pasos) => pasos.reduce((s, p) => s + p.dur, 0);

function hexRGB(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function rampa(hexes) {
  const c = hexes.map(hexRGB);
  return (v) => {
    v = lim(v) * (c.length - 1);
    const i = Math.min(c.length - 2, Math.floor(v)), f = v - i;
    return [0, 1, 2].map((k) => c[i][k] + (c[i + 1][k] - c[i][k]) * f);
  };
}
// Activaciones de la CNN (cian) e importancia del bosque (ámbar). La entrada se muestra en gris real.
export const rampaCNN = rampa(['#0E151C', '#0F3740', '#1F8794', '#43D9E8', '#EAFCFF']);
export const rampaRF = rampa(['#0E151C', '#3A2614', '#9A6029', '#F0A45A', '#FFF1DE']);
export const gris = (v) => { const g = 14 + lim(v) * 241; return [g, g, g]; };
export const APAGADO = [14, 21, 28];

// Texto en 3D (sprite que siempre mira a la cámara). poner() lo reescribe.
export class Rotulo {
  constructor({ alto = 0.3, color = T.tenue, peso = 500, mono = true, tam = 64, ancla = [0.5, 0.5] } = {}) {
    Object.assign(this, { alto, color, peso, mono, tam });
    this.cv = document.createElement('canvas');
    this.tex = new THREE.CanvasTexture(this.cv);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.tex.anisotropy = 4;
    this.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.tex, transparent: true, depthWrite: false }));
    this.sprite.center.set(...ancla);
    this.sprite.renderOrder = 10;
    this.txt = null;
  }
  poner(txt, color = this.color) {
    if (txt === this.txt && color === this.colorActual) return this;
    this.txt = txt; this.colorActual = color;
    const fuente = `${this.peso} ${this.tam}px ${this.mono ? FUENTE_MONO : FUENTE_SANS}`;
    const g = this.cv.getContext('2d');
    g.font = fuente;
    const w = Math.max(4, Math.ceil(g.measureText(txt).width) + 6);
    const h = Math.ceil(this.tam * 1.3);
    if (this.cv.width !== w || this.cv.height !== h) {
      this.cv.width = w; this.cv.height = h;
      this.tex.dispose(); // el tamaño cambió: la textura se vuelve a subir
    }
    g.clearRect(0, 0, w, h);
    g.font = fuente; g.fillStyle = color; g.textBaseline = 'middle';
    g.fillText(txt, 3, h / 2);
    this.tex.needsUpdate = true;
    this.sprite.scale.set((this.alto * w) / h, this.alto, 1);
    return this;
  }
}
export function rotulo(txt, opciones) { return new Rotulo(opciones).poner(txt); }

// Grilla de mapas cuadrados sobre el plano YZ (la red avanza en X). Todos comparten un atlas.
export function crearMapas({ n, cols, px, lado, sep = lado * 1.12, losa = true }) {
  const filas = Math.ceil(n / cols);
  const W = cols * px, H = filas * px;
  const data = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) data.set([...APAGADO, 255], i * 4);
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;

  const pos = new Float32Array(n * 12), uv = new Float32Array(n * 8), ind = [];
  const centros = [];
  const h = lado / 2;
  for (let c = 0; c < n; c++) {
    const col = c % cols, f = Math.floor(c / cols);
    const z = (col - (cols - 1) / 2) * sep, y = ((filas - 1) / 2 - f) * sep;
    centros.push(new THREE.Vector3(0, y, z));
    // esquinas: arriba-izq, arriba-der, abajo-der, abajo-izq (la fila 0 de la imagen va arriba)
    pos.set([0, y + h, z - h, 0, y + h, z + h, 0, y - h, z + h, 0, y - h, z - h], c * 12);
    const u0 = (col * px) / W, u1 = ((col + 1) * px) / W, v0 = (f * px) / H, v1 = ((f + 1) * px) / H;
    uv.set([u0, v0, u1, v0, u1, v1, u0, v1], c * 8);
    const b = c * 4;
    ind.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(ind);
  const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true });
  const mesh = new THREE.Mesh(geo, mat);
  const grupo = new THREE.Group();
  grupo.add(mesh);
  if (losa) {
    // cada mapa apoyado sobre una losa fina: le da volumen sin tapar el dato
    const losas = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.05, lado * 1.07, lado * 1.07),
      new THREE.MeshStandardMaterial({ color: T.panel, roughness: 0.75, metalness: 0.15, transparent: true }),
      n,
    );
    const m = new THREE.Matrix4();
    centros.forEach((p, c) => { m.setPosition(0.045, p.y, p.z); losas.setMatrixAt(c, m); });
    grupo.add(losas);
    grupo.userData.losas = losas;
  }
  const tile = (c) => ({ col: c % cols, fila: Math.floor(c / cols) });
  // Pinta el mapa c; color(i) recibe el índice de píxel (fila * px + col) y devuelve [r, g, b]
  function pintar(c, color) {
    const { col, fila } = tile(c);
    for (let i = 0; i < px * px; i++) {
      const R = fila * px + Math.floor(i / px), J = col * px + (i % px);
      const o = (R * W + J) * 4;
      const [r, g, b] = color(i);
      data[o] = r; data[o + 1] = g; data[o + 2] = b;
    }
    tex.needsUpdate = true;
  }
  return {
    grupo, mesh, mat, n, cols, filas, px, lado, sep, centros, pintar,
    ancho: cols * sep, alto: filas * sep,
    opacidad(a) { mat.opacity = a; if (grupo.userData.losas) grupo.userData.losas.material.opacity = a; grupo.visible = a > 0.01; },
    // centro en coordenadas locales del píxel (fila, col) del mapa c
    puntoPixel(c, fila, col) {
      const p = centros[c], d = lado / px;
      return new THREE.Vector3(-0.01, p.y + h - (fila + 0.5) * d, p.z - h + (col + 0.5) * d);
    },
  };
}

// Caja de alambre alrededor de un tensor, con su forma escrita abajo
export function cajaTensor(ancho, alto, prof, color = T.borde) {
  const caja = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(prof, alto, ancho)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
  );
  return caja;
}

// Segmentos con color RGBA por vértice (la transparencia codifica intensidad)
export function crearSegmentos(cantidad) {
  const pos = new Float32Array(cantidad * 6), col = new Float32Array(cantidad * 8);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 4));
  const lineas = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
  lineas.frustumCulled = false;
  return {
    lineas, pos, col,
    tramo(i, a, b) { pos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6); },
    color(i, c, alfa, alfa2 = alfa) { col.set([c.r, c.g, c.b, alfa, c.r, c.g, c.b, alfa2], i * 8); },
    listo(conPos = false) { geo.attributes.color.needsUpdate = true; if (conPos) { geo.attributes.position.needsUpdate = true; geo.computeBoundingSphere(); } },
    rango(n) { geo.setDrawRange(0, n * 2); },
  };
}
