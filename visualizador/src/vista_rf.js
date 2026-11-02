// Escena del Random Forest: un bosque de 200 árboles de decisión sobre el piso.
// Cada árbol se dibuja con su recorrido real para la imagen (raíz -> hoja), más las ramas que
// descartó en cada pregunta. Un árbol completo tiene miles de nodos: se muestra el camino que importa.
import * as THREE from 'three';
import { T, CLASES, Rotulo, rotulo, crearMapas, cajaTensor, crearSegmentos, rampaRF, gris,
  progresos, suave, lim, pc, num } from './comun.js';

export const PASOS_RF = [
  { id: 'in', nombre: 'Entrada', dur: 600, txt: 'El bosque recibe la imagen aplanada: 784 píxeles sueltos, de 0 a 255, sin saber cuáles son vecinos.' },
  { id: 'imp', nombre: 'Importancia de píxeles', dur: 700, txt: 'Cuánto ayudó cada píxel a separar las prendas durante el entrenamiento. Es fijo: no depende de esta imagen.' },
  { id: 'rec', nombre: 'Recorridos de decisión', dur: 3200, txt: 'Cada árbol pregunta si un píxel concreto supera un umbral y baja por la rama que corresponde, hasta una hoja.' },
  { id: 'vot', nombre: 'Votación', dur: 1600, txt: 'Cada hoja es un voto por una prenda. Los 200 votos convergen en el conteo.' },
  { id: 'may', nombre: 'Mayoría', dur: 800, txt: 'Gana la prenda más votada. El modelo reporta la proporción de votos, no una probabilidad calibrada.' },
];
export const VISTA_RF = { min: [-0.4, -3.5, -10], max: [31.4, 3.4, 10], dir: [-0.48, 0.55, 0.68] };

const PISO = -3.2;
const X = { in: 0, imp: 5.6, bosque: 9.8, votos: 31 };
const COLS = 20, SEP_F = 1.85, SEP_C = 1.0;
const AMBAR = new THREE.Color(T.rf), TENUE = new THREE.Color(T.tenue), CLARO = new THREE.Color('#FFE6C7'), BORDE = new THREE.Color(T.borde);
const MAXD = 64;

function circulo() {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const g = cv.getContext('2d');
  g.fillStyle = '#fff'; g.beginPath(); g.arc(32, 32, 28, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(cv);
  return t;
}

export function crearVistaRF(importancia, NA) {
  const grupo = new THREE.Group();
  const luz = new THREE.PointLight(T.rf, 80, 45, 2);
  luz.position.set(18, 4, 6);
  grupo.add(luz);

  // ---------- Entrada e importancia ----------
  const entrada = crearMapas({ n: 1, cols: 1, px: 28, lado: 4.2 });
  entrada.grupo.position.x = X.in;
  const imp = crearMapas({ n: 1, cols: 1, px: 28, lado: 3 });
  imp.grupo.position.x = X.imp;
  imp.pintar(0, (i) => rampaRF(Math.pow(importancia[i] / 255, 0.6)));
  for (const m of [entrada, imp]) {
    const c = cajaTensor(m.ancho + 0.12, m.alto + 0.12, 0.34); c.position.x = 0.03; m.grupo.add(c);
    grupo.add(m.grupo);
  }

  // ---------- Posición de cada árbol ----------
  const base = [], dir = [];
  for (let t = 0; t < NA; t++) {
    const f = Math.floor(t / COLS), c = t % COLS;
    base.push(new THREE.Vector3(X.bosque + f * SEP_F, PISO, (c - (COLS - 1) / 2) * SEP_C));
    const th = t * 2.39996; // ángulo dorado: cada árbol abre sus ramas en un plano distinto
    dir.push(new THREE.Vector3(Math.sin(th) * 0.55, 0, Math.cos(th)).normalize());
  }
  const filasBosque = Math.ceil(NA / COLS);
  const centroBosque = X.bosque + ((filasBosque - 1) * SEP_F) / 2;

  // Tronco fantasma de todos los árboles (siempre visible)
  const fantasma = crearSegmentos(NA);
  base.forEach((b, t) => { fantasma.tramo(t, b, b.clone().add(new THREE.Vector3(0, 0.45, 0))); fantasma.color(t, BORDE, 0.9); });
  fantasma.listo(true);
  grupo.add(fantasma.lineas);

  // Recorridos (capacidad para la profundidad máxima)
  const CAP = NA * (1 + 2 * MAXD);
  const ramas = crearSegmentos(CAP);
  const segArbol = new Uint16Array(CAP), segFrac = new Float32Array(CAP), segTipo = new Uint8Array(CAP);
  grupo.add(ramas.lineas);

  // Hojas: un punto por árbol
  const hojasPos = new Float32Array(NA * 3), hojasCol = new Float32Array(NA * 4);
  const geoHojas = new THREE.BufferGeometry();
  geoHojas.setAttribute('position', new THREE.BufferAttribute(hojasPos, 3));
  geoHojas.setAttribute('color', new THREE.BufferAttribute(hojasCol, 4));
  const mapaPunto = circulo();
  const hojas = new THREE.Points(geoHojas, new THREE.PointsMaterial({ size: 0.24, vertexColors: true, transparent: true, depthWrite: false, map: mapaPunto, alphaTest: 0.05 }));
  hojas.frustumCulled = false;
  grupo.add(hojas);

  // Volúmenes invisibles para pasar el mouse y elegir un árbol
  const golpe = new THREE.InstancedMesh(new THREE.BoxGeometry(0.7, 3.6, 0.7), new THREE.MeshBasicMaterial({ visible: false }), NA);
  { const m = new THREE.Matrix4(); base.forEach((b, t) => { m.setPosition(b.x, PISO + 1.8, b.z); golpe.setMatrixAt(t, m); }); }
  grupo.add(golpe);

  // Árbol elegido: sus nodos y las líneas a los píxeles que pregunta
  const nodosSelPos = new Float32Array((MAXD + 1) * 3);
  const geoSel = new THREE.BufferGeometry();
  geoSel.setAttribute('position', new THREE.BufferAttribute(nodosSelPos, 3));
  const nodosSel = new THREE.Points(geoSel, new THREE.PointsMaterial({ size: 0.17, color: '#FFE6C7', map: mapaPunto, alphaTest: 0.05, transparent: true }));
  nodosSel.frustumCulled = false;
  grupo.add(nodosSel);
  const aPixel = crearSegmentos(MAXD);
  grupo.add(aPixel.lineas);

  // ---------- Votos ----------
  const BASE_V = PISO, ALTO_MAX = 4.2, SEP_V = 0.86;
  const zV = (k) => (k - 4.5) * SEP_V;
  const votos = crearSegmentos(NA);
  grupo.add(votos.lineas);
  const barras = [], cuentas = [];
  for (let k = 0; k < 10; k++) {
    const geo = new THREE.BoxGeometry(0.4, 1, 0.4);
    geo.translate(0, 0.5, 0);
    const b = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: T.panel, roughness: 0.5, metalness: 0.2 }));
    b.position.set(X.votos, BASE_V, zV(k));
    b.scale.y = 0.02;
    b.userData.clase = k;
    grupo.add(b); barras.push(b);
    const et = rotulo(CLASES[k], { alto: 0.27 });
    et.sprite.material.rotation = Math.PI / 2;
    et.sprite.center.set(1, 0.5);
    et.sprite.position.set(X.votos, BASE_V - 0.3, zV(k));
    grupo.add(et.sprite);
    const c = new Rotulo({ alto: 0.3, color: T.texto });
    grupo.add(c.sprite); cuentas.push(c);
  }
  const veredicto = new Rotulo({ alto: 0.5, color: T.rf, peso: 600, mono: false });
  veredicto.sprite.position.set(X.votos, 2.35, 0);
  grupo.add(veredicto.sprite);

  // Títulos numerados
  [[X.in, 3.45, PASOS_RF[0]], [X.imp, 3.45, PASOS_RF[1]], [centroBosque, 2.4, PASOS_RF[2]], [X.votos, 3.3, PASOS_RF[3]]].forEach(([x, y, p]) => {
    const n = PASOS_RF.indexOf(p) + 1;
    const t = rotulo(`${String(n).padStart(2, '0')}  ${p.nombre}`, { alto: 0.5, color: T.texto, peso: 600, mono: false });
    t.sprite.position.set(x, y, 0);
    grupo.add(t.sprite);
  });
  [[X.in, 2.9, '784 píxeles'], [X.imp, 2.9, 'feature_importances_'], [centroBosque, 1.85, `${NA} árboles`]].forEach(([x, y, s]) => {
    const t = rotulo(s, { alto: 0.32 }); t.sprite.position.set(x, y, 0); grupo.add(t.sprite);
  });

  // ---------- Estado ----------
  let res = null, pix = null, caminos = null, sel = 0, version = 0, mayoria = 0, nSeg = 0;
  const hojaPunto = [];
  const nodosArbol = []; // por árbol: posiciones de sus nodos (para el árbol elegido)
  const cache = { clave: '', v: -1 };

  function construir() {
    let s = 0;
    hojaPunto.length = 0; nodosArbol.length = 0;
    for (let t = 0; t < NA; t++) {
      const { pasos } = caminos[t];
      const D = Math.min(pasos.length, MAXD);
      const H = Math.min(0.12, 2.9 / Math.max(1, D));
      const arriba = new THREE.Vector3(0, H, 0);
      let n = base[t].clone().add(new THREE.Vector3(0, 0.45, 0));
      const nodos = [n.clone()];
      ramas.tramo(s, base[t], n); segArbol[s] = t; segFrac[s] = 0; segTipo[s] = 2; s++;
      for (let d = 0; d < D; d++) {
        const w = 0.32 * Math.pow(0.8, d);
        const lado = pasos[d].izq ? -1 : 1;
        const sig = n.clone().add(arriba).addScaledVector(dir[t], lado * w);
        const desc = n.clone().addScaledVector(arriba, 0.75).addScaledVector(dir[t], -lado * w * 0.85);
        const frac = (d + 1) / (D + 1);
        ramas.tramo(s, n, sig); segArbol[s] = t; segFrac[s] = frac; segTipo[s] = 0; s++;
        ramas.tramo(s, n, desc); segArbol[s] = t; segFrac[s] = frac; segTipo[s] = 1; s++;
        n = sig; nodos.push(n.clone());
      }
      hojaPunto.push(n);
      nodosArbol.push(nodos);
      hojasPos.set([n.x, n.y, n.z], t * 3);
      const k = res.porArbol[t];
      votos.tramo(t, n, new THREE.Vector3(X.votos - 0.25, BASE_V + 0.12, zV(k)));
    }
    nSeg = s;
    ramas.rango(nSeg);
    ramas.listo(true);
    votos.listo(true);
    geoHojas.attributes.position.needsUpdate = true;
  }

  function actualizar(resultado, pixeles, caminoDe) {
    res = resultado; pix = pixeles; version++;
    caminos = Array.from({ length: NA }, (_, t) => caminoDe(t));
    mayoria = res.votos.indexOf(Math.max(...res.votos));
    construir();
    seleccionar(sel);
  }

  function seleccionar(t) {
    sel = t;
    if (!res) return;
    const nodos = nodosArbol[t], pasos = caminos[t].pasos;
    nodos.forEach((p, i) => nodosSelPos.set([p.x, p.y, p.z], i * 3));
    geoSel.setDrawRange(0, nodos.length - 1); // la hoja ya tiene su propio punto
    geoSel.attributes.position.needsUpdate = true;
    geoSel.computeBoundingSphere();
    // entrada en gris real, con los píxeles que pregunta este árbol en ámbar
    const preguntados = new Set(pasos.map((p) => p.pixel));
    entrada.pintar(0, (i) => (preguntados.has(i) ? [240, 164, 90] : gris(pix[i] / 255)));
    pasos.slice(0, MAXD).forEach((p, i) => {
      const f = Math.floor(p.pixel / 28), c = p.pixel % 28;
      aPixel.tramo(i, entrada.puntoPixel(0, f, c).add(new THREE.Vector3(X.in, 0, 0)), nodos[i]);
      aPixel.color(i, AMBAR, 0.08, 0.35);
    });
    aPixel.rango(Math.min(pasos.length, MAXD));
    aPixel.listo(true);
    cache.clave = '';
  }

  function aplicar(t) {
    if (!res) return;
    const u = progresos(PASOS_RF, t);
    entrada.opacidad(suave(u.in * 1.4));
    imp.opacidad(0.2 + 0.8 * suave(u.imp));
    const nV = u.vot >= 1 ? NA : Math.floor(u.vot * NA);
    const clave = `${u.rec.toFixed(4)}|${nV}|${u.may.toFixed(3)}|${sel}`;
    if (clave === cache.clave && cache.v === version) return;
    cache.clave = clave; cache.v = version;

    // crecimiento de cada árbol: la ola avanza fila por fila, alejándose de la entrada
    const crec = new Float32Array(NA);
    for (let a = 0; a < NA; a++) crec[a] = lim((u.rec - (a / NA) * 0.72) / 0.28);
    for (let s = 0; s < nSeg; s++) {
      const a = segArbol[s], g = crec[a];
      const vis = lim((g - segFrac[s]) * 12 + 1) * (g > 0 ? 1 : 0);
      const elegido = a === sel;
      if (segTipo[s] === 1) ramas.color(s, TENUE, vis * (elegido ? 0.55 : 0.22));
      else ramas.color(s, elegido ? CLARO : AMBAR, vis * (elegido ? 1 : segTipo[s] === 2 ? 0.55 : 0.8));
    }
    ramas.listo();
    nodosSel.visible = crec[sel] >= 1;
    aPixel.lineas.visible = crec[sel] >= 1;

    const cnt = new Array(10).fill(0);
    for (let a = 0; a < NA; a++) {
      const k = res.porArbol[a], llego = a < nV;
      if (llego) cnt[k]++;
      const c = k === mayoria ? AMBAR : TENUE;
      hojasCol.set([c.r, c.g, c.b, crec[a] >= 1 ? 1 : 0], a * 4);
      votos.color(a, c, 0, llego ? (k === mayoria ? 0.45 : 0.3) : 0);
    }
    geoHojas.attributes.color.needsUpdate = true;
    votos.listo();

    const fin = suave(u.may);
    barras.forEach((b, k) => {
      const p = cnt[k] / NA, alto = Math.max(0.02, p * ALTO_MAX);
      b.scale.y = alto;
      b.material.color.set(k === mayoria ? T.rf : '#5A4530');
      b.material.emissive.set(k === mayoria && fin > 0.5 ? '#4A2C0E' : '#000000');
      cuentas[k].poner(String(cnt[k]), k === mayoria ? T.texto : T.tenue);
      cuentas[k].sprite.position.set(X.votos, BASE_V + alto + 0.3, zV(k));
      cuentas[k].sprite.visible = cnt[k] > 0;
    });
    veredicto.poner(`${CLASES[mayoria]}: ${res.votos[mayoria]} de ${NA} votos`);
    veredicto.sprite.visible = fin > 0.05;
    veredicto.sprite.material.opacity = fin;
  }

  function hit(ray) {
    if (!res) return null;
    ray.params.Points.threshold = 0.09;
    if (nodosSel.visible) {
      const h = ray.intersectObject(nodosSel, false)[0];
      if (h && h.index < caminos[sel].pasos.length) {
        const p = caminos[sel].pasos[h.index];
        return { titulo: `Árbol ${sel + 1}, pregunta ${h.index + 1}`, detalle: condicion(p) };
      }
    }
    const hb = ray.intersectObjects(barras, false)[0];
    if (hb) { const k = hb.object.userData.clase; return { titulo: CLASES[k], detalle: `${res.votos[k]} de ${NA} votos (${pc(res.probs[k])} %)` }; }
    const ha = ray.intersectObject(golpe, false)[0];
    if (ha) {
      const a = ha.instanceId;
      return { titulo: `Árbol ${a + 1}`, detalle: `vota ${CLASES[res.porArbol[a]]} tras ${caminos[a].pasos.length} preguntas. Clic para verlo.`, arbol: a };
    }
    const hi = ray.intersectObjects([entrada.mesh, imp.mesh], false)[0];
    if (hi) return hi.object === imp.mesh
      ? { titulo: 'Importancia de cada píxel', detalle: 'más claro: más útil para separar prendas al entrenar' }
      : { titulo: 'Entrada', detalle: 'en ámbar, los píxeles que pregunta el árbol elegido' };
    return null;
  }

  return {
    grupo, actualizar, aplicar, hit, seleccionar,
    get seleccionado() { return sel; },
    get camino() { return caminos ? caminos[sel] : null; },
  };
}

export function condicion(p) {
  const f = Math.floor(p.pixel / 28), c = p.pixel % 28;
  const u = num((p.umbral + 0.5) / 255);
  return `¿píxel (${f}, ${c}) > ${u}? vale ${num(p.valor / 255)}: ${p.izq ? 'no' : 'sí'}`;
}
