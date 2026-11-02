// Escena de la CNN: entrada -> conv 32 -> pooling -> conv 64 -> pooling -> densa 128 -> softmax 10.
// Todo lo que se dibuja sale del forward real (cnn.js); el tiempo t solo decide cuánto se ve.
import * as THREE from 'three';
import { T, CLASES, Rotulo, rotulo, crearMapas, cajaTensor, crearSegmentos, rampaCNN, gris, APAGADO,
  progresos, suave, lim, pc, num } from './comun.js';

export const PASOS_CNN = [
  { id: 'in', nombre: 'Entrada', dur: 600, txt: 'La imagen en gris: 784 valores entre 0 (negro) y 1 (blanco).' },
  { id: 'c1', nombre: 'Convolución 1', dur: 3200, txt: '32 filtros de 3×3 recorren la imagen. En cada posición se multiplica, se suma el sesgo y ReLU anula lo negativo.' },
  { id: 'p1', nombre: 'Max pooling 1', dur: 900, txt: 'Cada zona de 2×2 se resume con su valor más alto: los mapas pasan de 28×28 a 14×14.' },
  { id: 'c2', nombre: 'Convolución 2', dur: 1400, txt: '64 filtros de 3×3 combinan los 32 mapas anteriores en patrones más complejos.' },
  { id: 'p2', nombre: 'Max pooling 2', dur: 900, txt: 'Otro resumen de 2×2: quedan 64 mapas de 7×7, 3.136 valores en total.' },
  { id: 'd1', nombre: 'Densa', dur: 900, txt: '128 neuronas combinan los 3.136 valores con pesos aprendidos y aplican ReLU.' },
  { id: 'out', nombre: 'Salida softmax', dur: 900, txt: '10 valores (logits) que softmax convierte en probabilidades que suman 1.' },
];
// Encuadre: caja que tiene que entrar en cámara y dirección desde la que se la mira
export const VISTA_CNN = { min: [-0.4, -3.4, -4.7], max: [25.4, 3.7, 4.7], dir: [-0.6, 0.36, 0.72] };

const X = { in: 0, c1: 5.8, p1: 10.2, c2: 14, p2: 17.8, d1: 21.2, out: 25 };
const CIAN = new THREE.Color(T.cnn);

export function crearVistaCNN(modelo) {
  const grupo = new THREE.Group();
  const luz = new THREE.PointLight(T.cnn, 70, 45, 2);
  luz.position.set(15, 6, 7);
  grupo.add(luz);

  // ---------- Capas ----------
  const capas = {
    in: crearMapas({ n: 1, cols: 1, px: 28, lado: 4.6 }),
    c1: crearMapas({ n: 32, cols: 8, px: 28, lado: 1.0, sep: 1.1 }),
    p1: crearMapas({ n: 32, cols: 8, px: 14, lado: 0.5, sep: 0.58 }),
    c2: crearMapas({ n: 64, cols: 8, px: 14, lado: 0.5, sep: 0.58 }),
    p2: crearMapas({ n: 64, cols: 8, px: 7, lado: 0.26, sep: 0.32 }),
  };
  const FORMAS = { in: '28 × 28 × 1', c1: '28 × 28 × 32', p1: '14 × 14 × 32', c2: '14 × 14 × 64', p2: '7 × 7 × 64', d1: '128', out: '10' };
  for (const [id, m] of Object.entries(capas)) {
    m.grupo.position.x = X[id];
    const caja = cajaTensor(m.ancho + 0.12, m.alto + 0.12, 0.34);
    caja.position.x = 0.03;
    m.grupo.add(caja);
    grupo.add(m.grupo);
  }

  // Densa: 128 neuronas
  const COLS_D = 16, SEP_D = 0.28;
  const neuronas = new THREE.InstancedMesh(new THREE.BoxGeometry(0.19, 0.19, 0.19), new THREE.MeshBasicMaterial({ color: 0xffffff }), 128);
  const posD = [];
  {
    const m = new THREE.Matrix4();
    for (let i = 0; i < 128; i++) {
      const p = new THREE.Vector3(X.d1, (3.5 - Math.floor(i / COLS_D)) * SEP_D, ((i % COLS_D) - 7.5) * SEP_D);
      posD.push(p); m.setPosition(p); neuronas.setMatrixAt(i, m); neuronas.setColorAt(i, new THREE.Color().setRGB(...APAGADO.map((v) => v / 255)));
    }
    grupo.add(neuronas);
    const caja = cajaTensor(COLS_D * SEP_D + 0.1, 8 * SEP_D + 0.1, 0.34);
    caja.position.set(X.d1, 0, 0);
    grupo.add(caja);
  }

  // Salida: 10 barras
  const BASE = -2.45, ALTO_MAX = 4, SEP_O = 0.86;
  const zO = (k) => (k - 4.5) * SEP_O;
  const barras = [], pcts = [];
  for (let k = 0; k < 10; k++) {
    const geo = new THREE.BoxGeometry(0.4, 1, 0.4);
    geo.translate(0, 0.5, 0);
    const b = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: T.panel, roughness: 0.5, metalness: 0.2 }));
    b.position.set(X.out, BASE, zO(k));
    b.scale.y = 0.02;
    b.userData.clase = k;
    grupo.add(b); barras.push(b);
    const et = rotulo(CLASES[k], { alto: 0.27, ancla: [1, 0.5] });
    et.sprite.position.set(X.out, BASE - 0.3, zO(k));
    et.sprite.material.rotation = Math.PI / 2;
    et.sprite.center.set(1, 0.5);
    grupo.add(et.sprite);
    const p = new Rotulo({ alto: 0.3, color: T.texto });
    grupo.add(p.sprite); pcts.push(p);
  }

  // Títulos numerados y forma de cada tensor
  PASOS_CNN.forEach((paso, k) => {
    const t = rotulo(`${String(k + 1).padStart(2, '0')}  ${paso.nombre}`, { alto: 0.5, color: T.texto, peso: 600, mono: false });
    t.sprite.position.set(X[paso.id], 3.45, 0);
    grupo.add(t.sprite);
    const f = rotulo(FORMAS[paso.id], { alto: 0.32 });
    f.sprite.position.set(X[paso.id], 2.9, 0);
    grupo.add(f.sprite);
  });

  // ---------- Conexiones reales: cada mapa con cada mapa siguiente (todas, con opacidad según activación) ----------
  const centro = (id, i) => {
    if (id === 'd1') return posD[i];
    if (id === 'out') return new THREE.Vector3(X.out, BASE + 0.6, zO(i));
    return capas[id].centros[i].clone().add(new THREE.Vector3(X[id], 0, 0));
  };
  const CONEX = [
    ['in', 'c1', 'todos', 1, 32], ['c1', 'p1', 'uno', 32, 32], ['p1', 'c2', 'todos', 32, 64],
    ['c2', 'p2', 'uno', 64, 64], ['p2', 'd1', 'todos', 64, 128], ['d1', 'out', 'todos', 128, 10],
  ];
  let totalSeg = 0;
  for (const [, , tipo, a, b] of CONEX) totalSeg += tipo === 'uno' ? a : a * b;
  const seg = crearSegmentos(totalSeg);
  const segA = new Uint16Array(totalSeg), segB = new Uint16Array(totalSeg), segC = new Uint8Array(totalSeg);
  {
    let s = 0;
    CONEX.forEach(([ia, ib, tipo, a, b], ci) => {
      for (let i = 0; i < a; i++) {
        for (let j = 0; j < (tipo === 'uno' ? 1 : b); j++) {
          const jj = tipo === 'uno' ? i : j;
          const pa = centro(ia, i).add(new THREE.Vector3(0.1, 0, 0)), pb = centro(ib, jj).add(new THREE.Vector3(-0.06, 0, 0));
          seg.tramo(s, pa, pb);
          segA[s] = i; segB[s] = jj; segC[s] = ci; s++;
        }
      }
    });
    seg.listo(true);
    grupo.add(seg.lineas);
  }

  // ---------- Kernel 3×3 que recorre la entrada ----------
  const dIn = 4.6 / 28;
  const marco = (lado) => {
    const h = lado / 2;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, h, -h), new THREE.Vector3(0, h, h), new THREE.Vector3(0, -h, h), new THREE.Vector3(0, -h, -h)]);
    return new THREE.LineLoop(g, new THREE.LineBasicMaterial({ color: T.cnn, depthTest: false }));
  };
  const kernel = marco(dIn * 3);
  const salidaK = marco(1.0 / 28 * 1.6);
  kernel.renderOrder = salidaK.renderOrder = 11;
  const campo = crearSegmentos(4);
  campo.lineas.renderOrder = 11;
  campo.lineas.material.depthTest = false;
  grupo.add(kernel, salidaK, campo.lineas);

  // ---------- Estado ----------
  let r = null, img = null, fMax = 0, version = 0;
  const topes = {};
  let intens = null;       // intensidad 0..1 de cada unidad, por capa
  const cache = { c1: -1, c2: -1, d1: -1, out: -1, lineas: '', v: -1 };

  const maximo = (a) => { let m = 1e-6; for (const v of a) if (v > m) m = v; return m; };
  function medias(datos, px, ch) {
    const m = new Array(ch).fill(0);
    for (let i = 0; i < px * px; i++) for (let c = 0; c < ch; c++) m[c] += datos[i * ch + c];
    const t = Math.max(1e-6, ...m);
    return m.map((v) => v / t);
  }
  function pintarCapa(id, datos, ch, tope, limite = Infinity) {
    const m = capas[id];
    for (let c = 0; c < m.n; c++) {
      m.pintar(c, (i) => (i < limite ? rampaCNN(Math.pow(datos[i * ch + c] / tope, 0.5)) : APAGADO));
    }
  }

  function actualizar(res, imagen) {
    r = res; img = imagen; version++;
    topes.a1 = maximo(r.a1); topes.p1 = maximo(r.p1); topes.a2 = maximo(r.a2); topes.p2 = maximo(r.p2); topes.d1 = maximo(r.d1);
    const m1 = medias(r.a1, 28, 32);
    fMax = m1.indexOf(1);
    intens = { in: [1], c1: m1, p1: medias(r.p1, 14, 32), c2: medias(r.a2, 14, 64), p2: medias(r.p2, 7, 64),
      d1: Array.from(r.d1, (v) => v / topes.d1), out: r.probs };
    capas.in.pintar(0, (i) => gris(img[i]));
    pintarCapa('p1', r.p1, 32, topes.p1);
    pintarCapa('p2', r.p2, 64, topes.p2);
    // alfa base de cada conexión
    alfaBase = new Float32Array(totalSeg);
    for (let s = 0; s < totalSeg; s++) {
      const [ia, ib, tipo] = CONEX[segC[s]];
      const k = Math.sqrt(intens[ia][segA[s]] * intens[ib][segB[s]]);
      // muchas líneas superpuestas: solo las conexiones entre unidades muy activas quedan visibles
      const n = CONEX[segC[s]][3] * CONEX[segC[s]][4];
      alfaBase[s] = tipo === 'uno' ? 0.12 + 0.45 * k : n <= 32 ? 0.08 + 0.4 * k : 0.5 * Math.pow(k, n > 2000 ? 5 : 3);
    }
    cache.c1 = cache.c2 = cache.d1 = cache.out = -1; cache.lineas = '';
  }
  let alfaBase = null;

  function aplicar(t) {
    if (!r) return;
    const u = progresos(PASOS_CNN, t);
    capas.in.opacidad(suave(u.in * 1.4));

    // Convolución 1: los 32 filtros se calculan a la vez en cada posición del kernel
    const L1 = u.c1 >= 1 ? 784 : Math.floor(u.c1 * 784);
    if (L1 !== cache.c1 || cache.v !== version) { pintarCapa('c1', r.a1, 32, topes.a1, L1); cache.c1 = L1; }
    capas.c1.opacidad(u.c1 > 0 ? 1 : 0.25);
    const barriendo = u.c1 > 0 && u.c1 < 1;
    kernel.visible = salidaK.visible = campo.lineas.visible = barriendo;
    if (barriendo) {
      const k = Math.min(783, L1), fi = Math.floor(k / 28), co = k % 28;
      kernel.position.copy(capas.in.puntoPixel(0, fi, co)).add(new THREE.Vector3(X.in - 0.03, 0, 0));
      salidaK.position.copy(capas.c1.puntoPixel(fMax, fi, co)).add(new THREE.Vector3(X.c1 - 0.03, 0, 0));
      const h = (dIn * 3) / 2;
      [[h, -h], [h, h], [-h, h], [-h, -h]].forEach(([dy, dz], i) => {
        campo.tramo(i, kernel.position.clone().add(new THREE.Vector3(0, dy, dz)), salidaK.position);
        campo.color(i, CIAN, 0.55, 0.9);
      });
      campo.listo(true);
    }

    // Pooling: los mapas aparecen al tamaño del mapa anterior y se achican a la mitad
    for (const [id, uu] of [['p1', u.p1], ['p2', u.p2]]) {
      const s = 1 + (1 - suave(uu)) * 1.0;
      capas[id].grupo.scale.set(1, s, s);
      capas[id].opacidad(uu > 0 ? 0.25 + 0.75 * suave(uu * 1.6) : 0.25);
    }

    const L2 = u.c2 >= 1 ? 196 : Math.floor(u.c2 * 196);
    if (L2 !== cache.c2 || cache.v !== version) { pintarCapa('c2', r.a2, 64, topes.a2, L2); cache.c2 = L2; }
    capas.c2.opacidad(u.c2 > 0 ? 1 : 0.25);

    const nD = Math.round(u.d1 * 128);
    if (nD !== cache.d1 || cache.v !== version) {
      const c = new THREE.Color();
      for (let i = 0; i < 128; i++) {
        const [rr, gg, bb] = i < nD ? rampaCNN(Math.pow(r.d1[i] / topes.d1, 0.7)) : APAGADO;
        neuronas.setColorAt(i, c.setRGB(rr / 255, gg / 255, bb / 255, THREE.SRGBColorSpace));
      }
      neuronas.instanceColor.needsUpdate = true;
      cache.d1 = nD;
    }

    const uo = suave(u.out);
    if (uo !== cache.out || cache.v !== version) {
      const top = r.probs.indexOf(Math.max(...r.probs));
      barras.forEach((b, k) => {
        const p = r.probs[k], alto = Math.max(0.02, p * ALTO_MAX * uo);
        b.scale.y = alto;
        b.material.color.set(k === top ? T.cnn : '#24434A');
        b.material.emissive.set(k === top && uo > 0.99 ? '#0E4A52' : '#000000');
        pcts[k].poner(`${pc(p, p >= 0.1 ? 0 : 1)}%`, k === top ? T.texto : T.tenue);
        pcts[k].sprite.position.set(X.out, BASE + alto + 0.3, zO(k));
        pcts[k].sprite.visible = u.out > 0.6 && (p >= 0.005 || k === top);
      });
      cache.out = uo;
    }

    // Conexiones: se encienden cuando se calcula la capa de destino
    const rev = [1, suave(u.c1 * 3), lim((u.p1 - 0.6) / 0.4), suave(u.c2 * 2), lim((u.p2 - 0.6) / 0.4), suave(u.d1 * 2), suave(u.out * 2)];
    const clave = rev.map((v) => v.toFixed(3)).join();
    if (clave !== cache.lineas || cache.v !== version) {
      for (let s = 0; s < totalSeg; s++) seg.color(s, CIAN, alfaBase[s] * rev[segC[s] + 1]);
      seg.listo();
      cache.lineas = clave;
    }
    cache.v = version;
  }

  // Lo que pasa en la posición actual del kernel, con los pesos reales del filtro más activo
  function info(t) {
    if (!r) return null;
    const u = progresos(PASOS_CNN, t);
    if (!(u.c1 > 0 && u.c1 < 1)) return null;
    const k = Math.min(783, Math.floor(u.c1 * 784)), fi = Math.floor(k / 28), co = k % 28;
    const { w, b } = modelo.c1;
    const parche = [], pesos = [];
    let suma = b[fMax];
    for (let ky = 0; ky < 3; ky++) {
      for (let kx = 0; kx < 3; kx++) {
        const y = fi + ky - 1, x = co + kx - 1;
        const v = y < 0 || y >= 28 || x < 0 || x >= 28 ? 0 : img[y * 28 + x];
        const wv = w[(ky * 3 + kx) * 32 + fMax];
        parche.push(v); pesos.push(wv); suma += v * wv;
      }
    }
    return { filtro: fMax, fila: fi, col: co, parche, pesos, sesgo: b[fMax], suma, relu: Math.max(0, suma), salida: r.a1[k * 32 + fMax] };
  }

  const NOMBRES = { in: 'Entrada', c1: 'Convolución 1', p1: 'Max pooling 1', c2: 'Convolución 2', p2: 'Max pooling 2' };
  const objetivos = [...Object.values(capas).map((m) => m.mesh), neuronas, ...barras];
  function hit(ray) {
    if (!r) return null;
    const h = ray.intersectObjects(objetivos, false)[0];
    if (!h) return null;
    if (h.object === neuronas) return { titulo: `Densa, neurona ${h.instanceId + 1}`, detalle: `activación ${num(r.d1[h.instanceId])} (ReLU)` };
    if (barras.includes(h.object)) { const k = h.object.userData.clase; return { titulo: CLASES[k], detalle: `logit ${num(r.z[k])}, softmax ${pc(r.probs[k], 2)} %` }; }
    const id = Object.keys(capas).find((c) => capas[c].mesh === h.object);
    const c = Math.floor(h.faceIndex / 2);
    if (id === 'in') return { titulo: 'Entrada', detalle: 'imagen de 28 × 28 en escala de grises' };
    return { titulo: `${NOMBRES[id]}, filtro ${c + 1}`, detalle: `activación media relativa ${num(intens[id][c])}` };
  }

  return { grupo, actualizar, aplicar, info, hit, get fMax() { return fMax; } };
}
