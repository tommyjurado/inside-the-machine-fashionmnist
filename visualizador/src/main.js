import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { construirModelo, adelante } from './cnn.js';
import { b64ABytes, construirBosque, votar, camino } from './rf.js';
import { T, CLASES, pc, num, miles, lim, duracion, progresos } from './comun.js';
import { crearVistaCNN, PASOS_CNN, VISTA_CNN } from './vista_cnn.js';
import { crearVistaRF, PASOS_RF, VISTA_RF } from './vista_rf.js';

const DATOS = window.DATOS;
const modelo = construirModelo(DATOS.pesos);
const NA = DATOS.rf.arboles;
const $ = (id) => document.getElementById(id);
const reducido = matchMedia('(prefers-reduced-motion: reduce)').matches;
const SEQ = { cnn: PASOS_CNN, rf: PASOS_RF };
const VISTAS = { cnn: VISTA_CNN, rf: VISTA_RF };
const dos = (n) => String(n).padStart(2, '0');

// ---------- Escena ----------
const cont = $('escena');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
cont.appendChild(renderer.domElement);
const escena = new THREE.Scene();
escena.background = new THREE.Color(T.fondo);
escena.fog = new THREE.Fog(T.fondo, 45, 100);
escena.add(new THREE.HemisphereLight('#DCECFF', T.fondo, 0.8));
const sol = new THREE.DirectionalLight('#FFFFFF', 1.2);
sol.position.set(-10, 16, 12);
escena.add(sol);
const piso = new THREE.GridHelper(110, 110, '#1E2A35', '#131B24');
piso.position.set(16, -3.2, 0);
escena.add(piso);

// Una cámara por modelo: en Comparar se ven las dos, cada una en su mitad
const camaras = {}, ctrls = {};
for (const k of ['cnn', 'rf']) {
  const cam = new THREE.PerspectiveCamera(38, 1, 0.1, 300);
  const c = new OrbitControls(cam, renderer.domElement);
  Object.assign(c, { enableDamping: true, dampingFactor: 0.08, minDistance: 5, maxDistance: 90, maxPolarAngle: Math.PI * 0.495, autoRotateSpeed: 0.5 });
  camaras[k] = cam; ctrls[k] = c;
}

let modo = 'cnn';
const activas = () => (modo === 'comp' ? ['cnn', 'rf'] : [modo]);
// Los controles de abajo tapan parte del lienzo: esa franja (res) no cuenta para el encuadre.
const reserva = () => document.querySelector('.controles').offsetHeight + 24;
// Regiones en píxeles con origen arriba (top). En Comparar: CNN arriba y bosque abajo,
// porque las dos arquitecturas son largas y aprovechan mejor el ancho.
function regiones() {
  const W = cont.clientWidth, H = cont.clientHeight, R = reserva();
  if (modo !== 'comp') return [{ k: modo, top: 0, w: W, h: H, res: R }];
  const arriba = Math.floor(Math.max(80, H - R) / 2);
  return [{ k: 'cnn', top: 0, w: W, h: arriba, res: 0 }, { k: 'rf', top: arriba, w: W, h: H - arriba, res: R }];
}
function regionEn(cx, cy) {
  const r = renderer.domElement.getBoundingClientRect();
  const x = cx - r.left, y = cy - r.top;
  const regs = regiones();
  const reg = regs.find((g) => y >= g.top && y < g.top + g.h) || regs[0];
  return { reg, ndc: new THREE.Vector2((x / reg.w) * 2 - 1, -((y - reg.top) / reg.h) * 2 + 1) };
}
// Los controles de órbita solo responden en la mitad donde empezó el gesto
function elegirControl(e) { const { reg } = regionEn(e.clientX, e.clientY); for (const k in ctrls) ctrls[k].enabled = k === reg.k; }
renderer.domElement.addEventListener('pointerdown', elegirControl, { capture: true });
renderer.domElement.addEventListener('wheel', elegirControl, { capture: true, passive: true });

// La proyección se corre hacia arriba para que la escena quede centrada en el área libre
// (setViewOffset sobre un cuadro virtual más alto que la región).
function proyeccion(cam, reg) {
  cam.aspect = reg.w / reg.h;
  if (reg.res > 0) cam.setViewOffset(reg.w, reg.h + reg.res, 0, reg.res, reg.w, reg.h);
  else cam.clearViewOffset();
  cam.updateProjectionMatrix();
}
// Encuadre: busca la distancia mínima a la que la caja de la arquitectura entra entera en el área libre
const camPrueba = new THREE.PerspectiveCamera(38, 1, 0.1, 300);
function encuadre(k) {
  const reg = regiones().find((r) => r.k === k) || { w: cont.clientWidth, h: cont.clientHeight, res: 0 };
  const v = VISTAS[k];
  const caja = new THREE.Box3(new THREE.Vector3(...v.min), new THREE.Vector3(...v.max));
  const mira = caja.getCenter(new THREE.Vector3());
  const dir = new THREE.Vector3(...v.dir).normalize();
  const esquinas = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => new THREE.Vector3(i & 1 ? v.max[0] : v.min[0], i & 2 ? v.max[1] : v.min[1], i & 4 ? v.max[2] : v.min[2]));
  proyeccion(camPrueba, reg);
  const abajo = -1 + (2 * reg.res) / reg.h + 0.04;
  const entra = (d) => {
    camPrueba.position.copy(mira).addScaledVector(dir, d);
    camPrueba.lookAt(mira);
    camPrueba.updateMatrixWorld();
    return esquinas.every((p) => { const q = p.clone().project(camPrueba); return Math.abs(q.x) < 0.95 && q.y < 0.93 && q.y > abajo && q.z < 1; });
  };
  let a = 4, b = 200;
  for (let i = 0; i < 30; i++) { const m = (a + b) / 2; if (entra(m)) b = m; else a = m; }
  return { mira, pos: mira.clone().addScaledVector(dir, b) };
}
const transiciones = {};
function irAVista(k, inmediato = false) {
  const { pos, mira } = encuadre(k);
  if (inmediato || reducido) { camaras[k].position.copy(pos); ctrls[k].target.copy(mira); delete transiciones[k]; return; }
  transiciones[k] = { p0: camaras[k].position.clone(), m0: ctrls[k].target.clone(), p1: pos, m1: mira, t0: performance.now() };
}
function moverCamaras(ahora) {
  for (const [k, tr] of Object.entries(transiciones)) {
    const s = lim((ahora - tr.t0) / 900), e = 1 - Math.pow(1 - s, 3);
    camaras[k].position.lerpVectors(tr.p0, tr.p1, e);
    ctrls[k].target.lerpVectors(tr.m0, tr.m1, e);
    if (s >= 1) delete transiciones[k];
  }
}
// Al cambiar el tamaño (ventana, paneles plegados) se reencuadra, salvo que el usuario haya movido la cámara
let movida = false;
// En Comparar: rótulo de cada franja y la línea que las separa
function ubicarRotulos() {
  const rb = regiones()[1];
  if (!rb) return;
  $('rot-b').style.top = `${rb.top + 14}px`;
  $('divisor').style.top = `${rb.top}px`;
}
function ajustar() {
  renderer.setSize(cont.clientWidth, cont.clientHeight);
  ubicarRotulos();
  if (!movida && camaras.cnn.position.lengthSq() > 0) for (const k of activas()) irAVista(k, true);
}
new ResizeObserver(ajustar).observe(cont);
renderer.domElement.addEventListener('pointerdown', () => { movida = true; });
renderer.domElement.addEventListener('wheel', () => { movida = true; }, { passive: true });

// ---------- Reloj de la secuencia: todo se dibuja a partir de t ----------
const reloj = { t: 0, play: false, vel: 1 };
const tFinal = () => Math.max(...activas().map((k) => duracion(SEQ[k])));
function bordes() {
  const s = new Set();
  for (const k of activas()) { let a = 0; for (const p of SEQ[k]) { a += p.dur; s.add(a); } }
  return [...s].sort((a, b) => a - b);
}
const ICONO_PLAY = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>';
const ICONO_PAUSA = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>';
function botonPlay() {
  const b = $('play');
  b.innerHTML = reloj.play ? `${ICONO_PAUSA}Pausa` : `${ICONO_PLAY}Reproducir`;
  b.setAttribute('aria-label', reloj.play ? 'Pausar' : 'Reproducir');
}
function reproducir(desdeCero = false) {
  if (desdeCero || reloj.t >= tFinal()) reloj.t = 0;
  reloj.play = true; botonPlay();
}
function pausar() { reloj.play = false; botonPlay(); }
function paso() { const b = bordes().find((x) => x > reloj.t + 1); reloj.t = b ?? tFinal(); pausar(); }
function alFinal() { reloj.t = tFinal(); pausar(); }

// ---------- Estado de la inferencia ----------
let vCNN = null, vRF = null, bosque = null;
let imgActual = new Float32Array(784), resCNN = null, resRF = null, pixRF = null, claseReal = null;
let tCNN = 0, tRF = 0;

function inferir(img, animar) {
  imgActual = img;
  const t0 = performance.now();
  resCNN = adelante(modelo, img);
  tCNN = performance.now() - t0;
  vCNN.actualizar(resCNN, img);
  inferirRF();
  paneles();
  if (animar && !reducido) reproducir(true); else alFinal();
}
function inferirRF() {
  if (!bosque) return;
  pixRF = Uint8Array.from(imgActual, (v) => Math.round(v * 255));
  const t0 = performance.now();
  resRF = votar(bosque, pixRF);
  tRF = performance.now() - t0;
  vRF.actualizar(resRF, pixRF, (t) => camino(bosque, t, pixRF));
}

// ---------- Panel de entrada ----------
const lienzo = $('lienzo');
const g = lienzo.getContext('2d', { willReadFrequently: true });
const chico = document.createElement('canvas');
chico.width = chico.height = 28;
const gc = chico.getContext('2d', { willReadFrequently: true });
function limpiarLienzo() { g.fillStyle = '#000'; g.fillRect(0, 0, 224, 224); }
function dibujarEnLienzo(img) {
  const d = new ImageData(28, 28);
  for (let i = 0; i < 784; i++) { const v = Math.round(img[i] * 255); d.data.set([v, v, v, 255], i * 4); }
  gc.putImageData(d, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(chico, 0, 0, 224, 224);
  g.imageSmoothingEnabled = true;
}
function desdeLienzo() {
  gc.imageSmoothingEnabled = true; gc.imageSmoothingQuality = 'high';
  gc.drawImage(lienzo, 0, 0, 28, 28);
  const d = gc.getImageData(0, 0, 28, 28).data;
  return Float32Array.from({ length: 784 }, (_, i) => d[i * 4] / 255);
}
function origen(texto, clase) {
  claseReal = clase ?? null;
  $('origen').textContent = texto;
  $('real').textContent = claseReal === null ? '—' : CLASES[claseReal];
}
function cargarPix(pix, texto, clase) {
  const img = Float32Array.from(pix, (v) => v / 255);
  dibujarEnLienzo(img);
  origen(texto, clase);
  inferir(img, true);
}

let dibujando = false, previo = null, pendiente = false;
function posicion(e) { const r = lienzo.getBoundingClientRect(); return [((e.clientX - r.left) * 224) / r.width, ((e.clientY - r.top) * 224) / r.height]; }
function trazo(e) {
  const p = posicion(e);
  g.strokeStyle = '#fff'; g.lineWidth = 17; g.lineCap = 'round'; g.lineJoin = 'round';
  g.beginPath(); g.moveTo(...(previo || p)); g.lineTo(...p); g.stroke();
  previo = p;
  if (!pendiente) { pendiente = true; requestAnimationFrame(() => { pendiente = false; inferir(desdeLienzo(), false); }); }
}
lienzo.addEventListener('pointerdown', (e) => { dibujando = true; previo = null; lienzo.setPointerCapture(e.pointerId); origen('dibujo a mano', null); trazo(e); });
lienzo.addEventListener('pointermove', (e) => { if (dibujando) trazo(e); });
lienzo.addEventListener('pointerup', () => { dibujando = false; previo = null; });
$('limpiar').onclick = () => { limpiarLienzo(); origen('dibujo a mano', null); inferir(new Float32Array(784), false); };

// 500 imágenes reales del conjunto de prueba (50 por prenda)
const MU = b64ABytes(DATOS.muestras.pix);
const muestra = (i) => MU.subarray(i * 784, (i + 1) * 784);
const porClase = CLASES.map((_, k) => DATOS.muestras.clase.map((c, i) => (c === k ? i : -1)).filter((i) => i >= 0));
const giro = new Array(10).fill(0);
function cargarMuestra(i) { cargarPix(muestra(i), `prueba n.º ${DATOS.muestras.idx[i]}`, DATOS.muestras.clase[i]); }
CLASES.forEach((n, k) => {
  const b = document.createElement('button');
  b.textContent = n;
  b.title = `Una imagen de ${n.toLowerCase()} del conjunto de prueba`;
  b.onclick = () => { const l = porClase[k]; cargarMuestra(l[giro[k]++ % l.length]); };
  $('categorias').appendChild(b);
});
$('aleatoria').onclick = () => cargarMuestra(Math.floor(Math.random() * DATOS.muestras.idx.length));
DATOS.casos.forEach((c) => {
  const b = document.createElement('button');
  b.textContent = c.nombre.replace(' · ', ': ');
  b.title = `Imagen ${c.idx} del conjunto de prueba (real: ${CLASES[c.clase]}). Es la de la slide 17.`;
  b.onclick = () => cargarPix(c.pix, `prueba n.º ${c.idx}`, c.clase);
  $('casos').appendChild(b);
});

// Una foto cualquiera: se encuadra en 24 px, se pasa a gris y se invierte si el fondo es claro (FashionMNIST tiene fondo negro)
$('archivo').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  const url = URL.createObjectURL(f);
  try {
    const im = new Image(); im.src = url; await im.decode();
    const lum = (d, i) => (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
    const m = document.createElement('canvas'); m.width = m.height = 64;
    const gm = m.getContext('2d', { willReadFrequently: true });
    gm.drawImage(im, 0, 0, 64, 64);
    const dm = gm.getImageData(0, 0, 64, 64).data;
    let borde = 0, nb = 0;
    for (let i = 0; i < 64; i++) for (const [x, y] of [[i, 0], [i, 63], [0, i], [63, i]]) { borde += lum(dm, (y * 64 + x) * 4); nb++; }
    borde /= nb;
    const c = document.createElement('canvas'); c.width = c.height = 28;
    const gg = c.getContext('2d', { willReadFrequently: true });
    const v = Math.round(borde * 255);
    gg.fillStyle = `rgb(${v},${v},${v})`; gg.fillRect(0, 0, 28, 28);
    const s = 24 / Math.max(im.width, im.height), w = im.width * s, h = im.height * s;
    gg.imageSmoothingQuality = 'high';
    gg.drawImage(im, (28 - w) / 2, (28 - h) / 2, w, h);
    const d = gg.getImageData(0, 0, 28, 28).data;
    let img = Float32Array.from({ length: 784 }, (_, i) => lum(d, i * 4));
    if (borde > 0.5) img = img.map((x) => 1 - x);
    let mn = 1, mx = 0;
    for (const x of img) { mn = Math.min(mn, x); mx = Math.max(mx, x); }
    if (mx - mn > 0.05) img = img.map((x) => (x - mn) / (mx - mn));
    dibujarEnLienzo(img);
    origen(`archivo ${f.name.length > 18 ? f.name.slice(0, 16) + '…' : f.name}`, null);
    inferir(img, true);
  } catch {
    $('origen').textContent = 'no se pudo leer la imagen';
  } finally {
    URL.revokeObjectURL(url);
  }
});

// Barras de clasificación (CNN arriba, bosque abajo)
CLASES.forEach((n, k) => {
  const f = document.createElement('div');
  f.className = 'barra'; f.id = 'barra-' + k;
  f.innerHTML = `<span class="nombre">${n}</span><span class="pistas"><i class="p-cnn"><u></u></i><i class="p-rf"><u></u></i></span><span class="vals"><b class="v-cnn c-cnn">–</b><b class="v-rf c-rf">–</b></span>`;
  $('barras').appendChild(f);
});

// ---------- Panel de análisis ----------
const top = (a) => a.indexOf(Math.max(...a));
const nParam = ['conv1', 'conv2', 'dense1', 'dense2'].reduce((s, k) => s + DATOS.pesos[k].w_shape.reduce((a, b) => a * b, 1) + DATOS.pesos[k].w_shape.at(-1), 0);
$('ex-cnn').textContent = `${pc(DATOS.exactitud, 2)} %`;
$('ex-rf').textContent = `${pc(DATOS.rf.exactitud, 2)} %`;
$('m-ex-cnn').textContent = `${pc(DATOS.exactitud, 2)} %`;
$('m-ex-rf').textContent = `${pc(DATOS.rf.exactitud, 2)} %`;
$('m-par').textContent = miles(nParam);
$('m-arb').textContent = miles(NA);
$('m-nod').textContent = miles(DATOS.rf.nodos);
$('formula-rf').innerHTML = `P̂(k) = votos<sub>k</sub> / ${NA}`;

function paneles() {
  if (!resCNN) return;
  const tc = top(resCNN.probs), tr = resRF ? top(resRF.probs) : -1;
  CLASES.forEach((_, k) => {
    const f = $('barra-' + k);
    f.querySelector('.p-cnn u').style.width = `${resCNN.probs[k] * 100}%`;
    f.querySelector('.v-cnn').textContent = `${pc(resCNN.probs[k])} %`;
    f.querySelector('.p-rf u').style.width = resRF ? `${resRF.probs[k] * 100}%` : '0';
    f.querySelector('.v-rf').textContent = resRF ? `${pc(resRF.probs[k])} %` : '–';
    f.classList.toggle('top-cnn', k === tc);
    f.classList.toggle('top-rf', k === tr);
    f.classList.toggle('real', k === claseReal);
  });

  // Resultado de cada arquitectura
  const marca = (k) => (claseReal === null ? '' : k === claseReal ? '<span class="marca-ok ok">acierta</span>' : '<span class="marca-ok mal">falla</span>');
  let html = '';
  if (activas().includes('cnn')) {
    html += `<div class="res-fila"><span class="modelo c-cnn">CNN</span><span class="clase">${CLASES[tc]}</span>${marca(tc)}<span class="metrica">softmax ${pc(resCNN.probs[tc])} %</span></div>`;
  }
  if (activas().includes('rf')) {
    html += resRF
      ? `<div class="res-fila"><span class="modelo c-rf">RF</span><span class="clase">${CLASES[tr]}</span>${marca(tr)}<span class="metrica">${resRF.votos[tr]} de ${NA} votos (${pc(resRF.probs[tr])} %)</span></div>`
      : `<div class="res-fila"><span class="modelo c-rf">RF</span><span class="metrica">descomprimiendo el bosque…</span></div>`;
  }
  const pie = [];
  if (modo === 'comp' && resRF) pie.push(tc === tr ? `Las dos arquitecturas eligen <b>${CLASES[tc]}</b>.` : `Difieren: la CNN elige <b>${CLASES[tc]}</b> y el bosque <b>${CLASES[tr]}</b>.`);
  pie.push(claseReal === null ? 'Sin clase real: la imagen no viene del conjunto de prueba.' : `Clase real: <b>${CLASES[claseReal]}</b>.`);
  html += `<div class="res-pie">${pie.join(' ')}</div>`;
  $('resultado').innerHTML = html;

  // CNN
  $('t-cnn').textContent = `${num(tCNN, 1)} ms`;
  const mx = Math.max(...resCNN.z);
  const orden = [...resCNN.z.keys()].sort((a, b) => resCNN.z[b] - resCNN.z[a]).slice(0, 4);
  $('logits').innerHTML = '<tr><th>Prenda</th><th>z</th><th>e<sup>z−máx</sup></th><th>p</th></tr>' +
    orden.map((k) => `<tr class="${k === tc ? 'gana' : ''}"><td>${CLASES[k]}</td><td>${num(resCNN.z[k])}</td><td>${num(Math.exp(resCNN.z[k] - mx), 3)}</td><td>${pc(resCNN.probs[k], 2)} %</td></tr>`).join('');

  // Random Forest
  if (resRF) {
    $('t-rf').textContent = `${num(tRF, 1)} ms`;
    const conVotos = [...resRF.votos.keys()].filter((k) => resRF.votos[k] > 0).sort((a, b) => resRF.votos[b] - resRF.votos[a]);
    $('votos').innerHTML = '<tr><th>Prenda</th><th>votos</th><th>proporción</th></tr>' +
      conVotos.map((k) => `<tr class="${k === tr ? 'gana' : ''}"><td>${CLASES[k]}</td><td>${resRF.votos[k]}</td><td>${pc(resRF.probs[k])} %</td></tr>`).join('');
    panelArbol();
  } else {
    $('votos').innerHTML = '';
    $('arbol').innerHTML = '<div class="vacio">El bosque se está descomprimiendo.</div>';
  }
  leyendaClave = '';
}

const CHEVRON = (d) => `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="${d}"/></svg>`;
function panelArbol() {
  const c = vRF.camino, a = vRF.seleccionado;
  if (!c) return;
  const filas = c.pasos.map((p) => {
    const f = Math.floor(p.pixel / 28), co = p.pixel % 28;
    return `<li><span>píxel (${f}, ${co}) &gt; ${num((p.umbral + 0.5) / 255)}? vale <b>${num(p.valor / 255)}</b>: ${p.izq ? 'no' : 'sí'}</span></li>`;
  }).join('');
  $('arbol').innerHTML =
    `<div class="arbol-cab"><span>Árbol <b>${a + 1}</b> de ${NA}: vota <b class="c-rf">${CLASES[c.clase]}</b> tras ${c.pasos.length} preguntas</span>` +
    `<div class="grupo"><button id="a-ant" aria-label="Árbol anterior">${CHEVRON('M10 3 5 8l5 5')}</button><button id="a-sig" aria-label="Árbol siguiente">${CHEVRON('M6 3l5 5-5 5')}</button><button id="a-azar">Al azar</button></div></div>` +
    `<ol class="preguntas">${filas}<li class="hoja"><span>hoja: ${CLASES[c.clase]}</span></li></ol>` +
    '<p class="nota">Hacé clic en cualquier árbol del bosque para ver sus preguntas. En la entrada se marcan en ámbar los píxeles que mira.</p>';
  const ir = (t) => { vRF.seleccionar(t); panelArbol(); };
  $('a-ant').onclick = () => ir((a + NA - 1) % NA);
  $('a-sig').onclick = () => ir((a + 1) % NA);
  $('a-azar').onclick = () => ir(Math.floor(Math.random() * NA));
}

// Convolución en vivo: los números reales del filtro más activo en la posición del kernel
let convClave = null;
function convVivo() {
  const inf = vCNN.info(reloj.t);
  const clave = inf ? `${inf.fila},${inf.col}` : `nada${vCNN.fMax}`;
  if (clave === convClave) return;
  convClave = clave;
  const el = $('conv-vivo');
  if (!inf) {
    el.innerHTML = `<div class="vacio">Durante el paso 02 se ve acá la cuenta del filtro ${vCNN.fMax + 1} (el más activo) en cada posición del kernel.</div>`;
    return;
  }
  const m = (vals) => `<div class="m3">${vals.map((v) => `<span>${num(v)}</span>`).join('')}</div>`;
  el.innerHTML = `<div class="vivo">Filtro ${inf.filtro + 1}, posición (${inf.fila}, ${inf.col})` +
    `<div class="matrices">${m(inf.parche)}<span class="op">⊙</span>${m(inf.pesos)}</div>` +
    `<div class="cuenta">Σ x·w + b = ${num(inf.suma - inf.sesgo, 3)} + ${num(inf.sesgo, 3)} = ${num(inf.suma, 3)}</div>` +
    `<div class="cuenta">ReLU: ${num(inf.relu, 3)}</div></div>`;
}

// ---------- Regla de pasos y leyenda ----------
function armarReglas() {
  const r = $('reglas');
  r.innerHTML = '';
  for (const k of activas()) {
    const fila = document.createElement('div');
    fila.className = `regla ${k}`;
    let ini = 0;
    SEQ[k].forEach((p, i) => {
      const b = document.createElement('button');
      b.style.setProperty('--d', p.dur);
      b.innerHTML = `<span class="pista"><span></span></span><span class="et"><i>${dos(i + 1)}</i><span>${p.nombre}</span></span>`;
      b.title = `${p.nombre}: ${p.txt}`;
      const t0 = ini;
      b.onclick = () => { reloj.t = t0; reproducir(); };
      fila.appendChild(b);
      ini += p.dur;
    });
    r.appendChild(fila);
  }
}
let leyendaClave = '';
function actualizarReglas() {
  const actuales = [];
  activas().forEach((k, ri) => {
    const fila = $('reglas').children[ri];
    if (!fila) return;
    const u = progresos(SEQ[k], reloj.t);
    let actual = SEQ[k].findIndex((p) => u[p.id] < 1);
    if (actual < 0) actual = SEQ[k].length - 1;
    SEQ[k].forEach((p, i) => {
      const b = fila.children[i];
      b.querySelector('.pista span').style.transform = `scaleX(${u[p.id]})`;
      b.classList.toggle('actual', i === actual);
    });
    actuales.push([k, actual]);
  });
  const clave = actuales.map((a) => a.join(':')).join();
  if (clave === leyendaClave) return;
  leyendaClave = clave;
  $('leyenda-paso').innerHTML = actuales.map(([k, i]) => {
    const p = SEQ[k][i];
    return `<div><span class="num c-${k}">${dos(i + 1)}</span><b>${p.nombre}</b>${p.txt}</div>`;
  }).join('');
}

// ---------- Modos y controles ----------
function setModo(m) {
  modo = m;
  document.body.dataset.modo = m;
  document.querySelectorAll('#modos button').forEach((b) => b.classList.toggle('on', b.dataset.modo === m));
  document.querySelectorAll('[data-en]').forEach((el) => { el.hidden = !el.dataset.en.split(' ').includes(m); });
  $('rot-a').hidden = $('rot-b').hidden = $('divisor').hidden = m !== 'comp';
  armarReglas();
  ubicarRotulos();
  leyendaClave = '';
  if (reloj.t > tFinal()) reloj.t = tFinal();
  movida = false;
  for (const k of activas()) irAVista(k);
  paneles();
}
document.querySelectorAll('#modos button').forEach((b) => { b.onclick = () => setModo(b.dataset.modo); });
document.querySelectorAll('#velocidad button').forEach((b) => {
  b.onclick = () => { reloj.vel = +b.dataset.v; document.querySelectorAll('#velocidad button').forEach((x) => x.classList.toggle('on', x === b)); };
});
$('play').onclick = () => (reloj.play ? pausar() : reproducir());
$('paso').onclick = paso;
$('reiniciar').onclick = () => reproducir(true);
$('rotar').onclick = () => {
  const on = !ctrls.cnn.autoRotate;
  for (const k in ctrls) ctrls[k].autoRotate = on;
  $('rotar').classList.toggle('on', on);
  $('rotar').setAttribute('aria-pressed', on);
};
$('vista').onclick = () => { movida = false; for (const k of activas()) irAVista(k); };
document.querySelectorAll('[data-panel]').forEach((b) => {
  b.onclick = () => $('app').classList.toggle(`sin-${b.dataset.panel}`);
});
window.addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea, button, label')) return;
  if (e.code === 'Space') { e.preventDefault(); reloj.play ? pausar() : reproducir(); }
  if (e.code === 'ArrowRight') { e.preventDefault(); paso(); }
});

// ---------- Mouse sobre la escena: datos técnicos y elegir árbol ----------
const ray = new THREE.Raycaster();
const tip = $('tip');
function golpe(e) {
  const { reg, ndc } = regionEn(e.clientX, e.clientY);
  ray.setFromCamera(ndc, camaras[reg.k]);
  if (reg.k === 'cnn') return vCNN.hit(ray);
  return resRF ? vRF.hit(ray) : null;
}
renderer.domElement.addEventListener('pointermove', (e) => {
  if (e.buttons) { tip.style.display = 'none'; return; }
  const h = golpe(e);
  if (!h) { tip.style.display = 'none'; renderer.domElement.style.cursor = ''; return; }
  tip.innerHTML = `<b>${h.titulo}</b><span>${h.detalle}</span>`;
  tip.style.display = 'block';
  tip.style.left = Math.min(window.innerWidth - 310, e.clientX + 14) + 'px';
  tip.style.top = e.clientY + 14 + 'px';
  renderer.domElement.style.cursor = h.arbol !== undefined ? 'pointer' : '';
});
renderer.domElement.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
let abajo = null;
renderer.domElement.addEventListener('pointerdown', (e) => { abajo = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!abajo || Math.hypot(e.clientX - abajo[0], e.clientY - abajo[1]) > 4) return;
  const h = golpe(e);
  if (h && h.arbol !== undefined) { vRF.seleccionar(h.arbol); panelArbol(); }
});

// ---------- Bucle ----------
let antes = performance.now();
function bucle(ahora) {
  requestAnimationFrame(bucle);
  const dt = Math.min(64, ahora - antes);
  antes = ahora;
  if (reloj.play) {
    reloj.t += dt * reloj.vel;
    if (reloj.t >= tFinal()) { reloj.t = tFinal(); pausar(); }
  }
  const act = activas();
  if (act.includes('cnn')) { vCNN.aplicar(reloj.t); convVivo(); }
  if (act.includes('rf') && resRF) vRF.aplicar(reloj.t);
  actualizarReglas();
  moverCamaras(ahora);
  renderer.setScissorTest(true);
  for (const r of regiones()) {
    if (r.w < 2 || r.h < 2) continue;
    ctrls[r.k].update();
    proyeccion(camaras[r.k], r);
    const cam = camaras[r.k];
    vCNN.grupo.visible = r.k === 'cnn';
    vRF.grupo.visible = r.k === 'rf';
    // la niebla acompaña a la distancia de cada cámara: funde la retícula del piso sin apagar la escena
    const d = cam.position.distanceTo(ctrls[r.k].target);
    escena.fog.near = d * 0.9; escena.fog.far = d * 2.4;
    const y = cont.clientHeight - r.top - r.h; // WebGL cuenta desde abajo
    renderer.setViewport(0, y, r.w, r.h);
    renderer.setScissor(0, y, r.w, r.h);
    renderer.render(escena, cam);
  }
}

// ---------- Arranque ----------
async function cargarBosque() {
  const flujo = new Blob([b64ABytes(DATOS.rf.datos)]).stream().pipeThrough(new DecompressionStream('gzip'));
  const crudo = new Uint8Array(await new Response(flujo).arrayBuffer());
  bosque = construirBosque(crudo, DATOS.rf);
  $('est-rf').textContent = 'listo';
  $('led-rf').classList.remove('esperando');
  document.querySelectorAll('#modos button').forEach((b) => { b.disabled = false; });
  inferirRF();
  paneles();
}

async function iniciar() {
  // los rótulos 3D se dibujan en canvas: hace falta que las fuentes estén cargadas antes
  const fuentes = Promise.all([document.fonts.load('600 32px "Inter Variable"'), document.fonts.load('500 32px "JetBrains Mono Variable"')]);
  await Promise.race([fuentes, new Promise((r) => setTimeout(r, 1500))]).catch(() => {});
  vCNN = crearVistaCNN(modelo);
  vRF = crearVistaRF(b64ABytes(DATOS.rf.importancia), NA);
  escena.add(vCNN.grupo, vRF.grupo);
  ajustar();
  irAVista('cnn', true); irAVista('rf', true);
  limpiarLienzo();
  botonPlay();
  setModo('cnn');
  const c = DATOS.casos[0];
  cargarPix(c.pix, `prueba n.º ${c.idx}`, c.clase);
  requestAnimationFrame(bucle);
  cargarBosque().catch((err) => { $('est-rf').textContent = 'no se pudo cargar'; $('led-rf').classList.remove('esperando'); console.error(err); });
}
iniciar();

window.__viz = {
  setModo, reproducir, pausar, paso, reloj, cargarMuestra,
  get cnn() { return resCNN; }, get rf() { return resRF; }, get bosque() { return bosque; }, get vRF() { return vRF; }, get vCNN() { return vCNN; },
};
