import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { construirModelo, adelante } from './cnn.js';
import { b64ABytes, construirBosque, votar, camino } from './rf.js';

const DATOS = window.DATOS;
const pc = (p, d = 1) => (p * 100).toFixed(d).replace('.', ',');
const modelo = construirModelo(DATOS.pesos);
const CLASES = ['Remera', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota'];

// ---------- Descripción de las capas (las usa la escena y el panel lateral) ----------
const CAPAS = [
  { id: 'in', nombre: 'Entrada', forma: '28×28×1', x: 0, txt: 'La imagen en gris: 784 números entre 0 (negro) y 1 (blanco).' },
  { id: 'c1', nombre: 'Convolución 1', forma: '28×28×32', x: 7, txt: '32 filtros de 3×3 recorren la imagen. Cada uno resalta un patrón distinto (bordes, contornos) y ReLU borra lo negativo.' },
  { id: 'p1', nombre: 'Max pooling 1', forma: '14×14×32', x: 13, txt: 'Se queda con el valor más fuerte de cada zona de 2×2. La imagen se achica a la mitad.' },
  { id: 'c2', nombre: 'Convolución 2', forma: '14×14×64', x: 18.5, txt: '64 filtros que combinan los patrones simples en formas más complejas (cuellos, suelas, mangas).' },
  { id: 'p2', nombre: 'Max pooling 2', forma: '7×7×64', x: 24, txt: 'Otro achique: queda un resumen de 7×7 por filtro, 3.136 números en total.' },
  { id: 'd1', nombre: 'Densa (128)', forma: '128', x: 29, txt: '128 neuronas combinan los 3.136 números. Cada una suma todo con pesos aprendidos.' },
  { id: 'out', nombre: 'Salida (softmax)', forma: '10', x: 34, txt: '10 valores que softmax convierte en probabilidades: suman 100%.' },
];

// ---------- Escena ----------
const cont = document.getElementById('escena');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
cont.appendChild(renderer.domElement);
const escena = new THREE.Scene();
escena.background = new THREE.Color(0x05070d);
escena.fog = new THREE.Fog(0x05070d, 45, 95);
const cnnG = new THREE.Group(); escena.add(cnnG);   // todo lo de la CNN
const rfG = new THREE.Group(); escena.add(rfG);     // todo lo del Random Forest
const cam = new THREE.PerspectiveCamera(45, 1, 0.1, 300);
const VISTA = { pos: [-13, 9, 27], mira: [19, -0.5, 0] };
cam.position.set(...VISTA.pos);
const orbit = new OrbitControls(cam, renderer.domElement);
orbit.target.set(...VISTA.mira);
orbit.enableDamping = true;
orbit.autoRotateSpeed = 0.6;

function ajustar() {
  const w = cont.clientWidth, h = cont.clientHeight;
  renderer.setSize(w, h);
  cam.aspect = w / h;
  cam.updateProjectionMatrix();
}
window.addEventListener('resize', ajustar);

// Mapa de color: negro azulado -> turquesa -> blanco
function color(v) {
  v = Math.min(1, Math.max(0, v));
  if (v < 0.6) { const t = v / 0.6; return [8 + 0 * t, 12 + 200 * t, 24 + 176 * t]; }
  const t = (v - 0.6) / 0.4;
  return [8 + 247 * t, 212 + 43 * t, 200 + 55 * t];
}

const ESC = 0.04; // unidades por píxel
const mapas = {};   // id -> [{mesh, tex, data}]
const todosLosPlanos = [];

function crearPlano(lado, px, nombre) {
  const data = new Uint8Array(px * px * 4);
  const tex = new THREE.DataTexture(data, px, px, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.flipY = false;
  tex.needsUpdate = true;
  const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(lado, lado), mat);
  mesh.rotation.y = Math.PI / 2; // el plano mira hacia -X / +X
  mesh.userData.nombre = nombre;
  todosLosPlanos.push(mesh);
  return { mesh, tex, data, px };
}

function crearGrilla(id, cantidad, columnas, px, escala, nombre) {
  const lado = px * escala;
  const celda = lado * 1.12;
  const filas = Math.ceil(cantidad / columnas);
  const grupo = new THREE.Group();
  const lista = [];
  for (let i = 0; i < cantidad; i++) {
    const p = crearPlano(lado, px, `${nombre} · filtro ${i + 1}`);
    const c = i % columnas, f = Math.floor(i / columnas);
    p.mesh.position.set(0, ((filas - 1) / 2 - f) * celda, (c - (columnas - 1) / 2) * celda);
    grupo.add(p.mesh);
    lista.push(p);
  }
  grupo.position.x = CAPAS.find((c) => c.id === id).x;
  cnnG.add(grupo);
  mapas[id] = lista;
  return { grupo, alto: filas * celda, ancho: columnas * celda };
}

const gEntrada = crearGrilla('in', 1, 1, 28, 0.1, 'Entrada');
const gC1 = crearGrilla('c1', 32, 8, 28, ESC, 'Convolución 1');
const gP1 = crearGrilla('p1', 32, 8, 14, ESC, 'Pooling 1');
const gC2 = crearGrilla('c2', 64, 8, 14, ESC, 'Convolución 2');
const gP2 = crearGrilla('p2', 64, 8, 7, ESC, 'Pooling 2');

// Neuronas de la capa densa: 128 cubitos
const geoCubo = new THREE.BoxGeometry(0.22, 0.22, 0.22);
const matCubo = new THREE.MeshBasicMaterial({ color: 0xffffff });
const neuronas = new THREE.InstancedMesh(geoCubo, matCubo, 128);
const posD1 = [];
{
  const m = new THREE.Matrix4();
  for (let i = 0; i < 128; i++) {
    const c = i % 16, f = Math.floor(i / 16);
    const y = (3.5 - f) * 0.5, z = (c - 7.5) * 0.5;
    posD1.push(new THREE.Vector3(CAPAS[5].x, y, z));
    m.setPosition(CAPAS[5].x, y, z);
    neuronas.setMatrixAt(i, m);
    neuronas.setColorAt(i, new THREE.Color(0x0a1220));
  }
  cnnG.add(neuronas);
}

// Salida: 10 columnas cuya altura sigue a la probabilidad
const salidas = [];
const posOut = [];
function etiqueta(texto, tam = 64, col = '#cfe8ff') {
  const cv = document.createElement('canvas');
  cv.width = 512; cv.height = 128;
  const g = cv.getContext('2d');
  g.font = `bold ${tam}px ui-monospace, Consolas, monospace`;
  g.fillStyle = col; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(texto, 256, 64);
  const t = new THREE.CanvasTexture(cv);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthTest: false }));
  s.scale.set(4, 1, 1);
  return s;
}
for (let k = 0; k < 10; k++) {
  const z = (k - 4.5) * 0.95;
  const geo = new THREE.BoxGeometry(0.5, 1, 0.5);
  geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshBasicMaterial({ color: 0x1a2438 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(CAPAS[6].x, -2.2, z);
  mesh.scale.y = 0.05;
  cnnG.add(mesh);
  const et = etiqueta(CLASES[k], 56);
  et.scale.set(2.2, 0.55, 1);
  et.position.set(CAPAS[6].x, -2.75, z);
  cnnG.add(et);
  const pct = etiqueta('0%', 56, '#ffffff');
  pct.scale.set(1.4, 0.35, 1);
  cnnG.add(pct);
  salidas.push({ mesh, mat, pct, cv: pct.material.map.image });
  posOut.push(new THREE.Vector3(CAPAS[6].x, -2.2, z));
}

// Rótulos de cada capa
const altoCartel = 4.6;
CAPAS.forEach((c) => {
  const s = etiqueta(c.nombre, 54, '#9fd8ff');
  s.scale.set(4.2, 1.05, 1);
  s.position.set(c.x, altoCartel, 0);
  cnnG.add(s);
  const s2 = etiqueta(c.forma, 48, '#5b6b86');
  s2.scale.set(3.4, 0.85, 1);
  s2.position.set(c.x, altoCartel - 0.65, 0);
  cnnG.add(s2);
});

// ---------- Conexiones (muestra de líneas entre capas) ----------
function centros(id) {
  if (id === 'd1') return posD1;
  if (id === 'out') return posOut;
  return mapas[id].map((p) => p.mesh.getWorldPosition(new THREE.Vector3()));
}
// Hace falta que las matrices del mundo estén actualizadas antes de pedir posiciones
escena.updateMatrixWorld(true);

const conexiones = []; // {lineas, origen, destino, pares, idCapaDestino}
function conectar(idA, idB, tipo, cantidad, capaDestinoIdx) {
  const A = centros(idA), B = centros(idB);
  const pares = [];
  if (tipo === 'uno') {
    for (let i = 0; i < A.length; i++) pares.push([i, i]);
  } else if (tipo === 'todos') {
    for (let i = 0; i < A.length; i++) for (let j = 0; j < B.length; j++) pares.push([i, j]);
  } else {
    for (let n = 0; n < cantidad; n++) pares.push([Math.floor(Math.random() * A.length), Math.floor(Math.random() * B.length)]);
  }
  const pos = new Float32Array(pares.length * 6);
  pares.forEach(([i, j], n) => {
    pos.set([A[i].x + 0.02, A[i].y, A[i].z, B[j].x - 0.02, B[j].y, B[j].z], n * 6);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const col = new Float32Array(pares.length * 6);
  const attr = new THREE.BufferAttribute(col, 3);
  geo.setAttribute('color', attr);
  const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false });
  const lineas = new THREE.LineSegments(geo, mat);
  cnnG.add(lineas);
  conexiones.push({ lineas, mat, attr, pares, idA, idB, capaDestinoIdx });
}
conectar('in', 'c1', 'todos', 0, 1);
conectar('c1', 'p1', 'uno', 0, 2);
conectar('p1', 'c2', 'muestra', 450, 3);
conectar('c2', 'p2', 'uno', 0, 4);
conectar('p2', 'd1', 'muestra', 700, 5);
conectar('d1', 'out', 'todos', 0, 6);

// ---------- Actualización con una nueva imagen ----------
let ultimo = null;           // resultado del forward
let revelado = [1, 1, 1, 1, 1, 1, 1]; // 0..1 por capa (para la animación de "recorrido")

function pintarMapas(id, datos, px, canales, tope) {
  const lista = mapas[id];
  for (let c = 0; c < canales; c++) {
    const { data, tex } = lista[c];
    for (let i = 0; i < px * px; i++) {
      const v = Math.pow(Math.min(1, datos[i * canales + c] / tope), 0.7);
      const [r, g, b] = color(v);
      data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255;
    }
    tex.needsUpdate = true;
  }
}
const maximo = (arr) => { let m = 1e-6; for (const v of arr) if (v > m) m = v; return m; };

// Intensidad de cada "unidad" (un mapa o una neurona) usada para iluminar las líneas
function intensidades(id, r) {
  const norm = (a) => { const m = maximo(a); return Array.from(a, (v) => v / m); };
  if (id === 'in') return [1];
  if (id === 'd1') return norm(r.d1);
  if (id === 'out') return r.probs;
  const [datos, px, ch] = { c1: [r.a1, 28, 32], p1: [r.p1, 14, 32], c2: [r.a2, 14, 64], p2: [r.p2, 7, 64] }[id];
  const medias = new Array(ch).fill(0);
  for (let i = 0; i < px * px; i++) for (let c = 0; c < ch; c++) medias[c] += datos[i * ch + c];
  return norm(medias);
}

function actualizar(img) {
  const r = adelante(modelo, img);
  ultimo = r;
  // Entrada
  const e = mapas.in[0];
  for (let i = 0; i < 784; i++) {
    const [rr, gg, bb] = color(Math.pow(img[i], 0.8));
    e.data[i * 4] = rr; e.data[i * 4 + 1] = gg; e.data[i * 4 + 2] = bb; e.data[i * 4 + 3] = 255;
  }
  e.tex.needsUpdate = true;
  pintarMapas('c1', r.a1, 28, 32, maximo(r.a1));
  pintarMapas('p1', r.p1, 14, 32, maximo(r.p1));
  pintarMapas('c2', r.a2, 14, 64, maximo(r.a2));
  pintarMapas('p2', r.p2, 7, 64, maximo(r.p2));

  // Neuronas
  const mx = maximo(r.d1);
  const c = new THREE.Color();
  for (let i = 0; i < 128; i++) {
    const [rr, gg, bb] = color(Math.pow(r.d1[i] / mx, 0.7));
    c.setRGB(rr / 255, gg / 255, bb / 255);
    neuronas.setColorAt(i, c);
  }
  neuronas.instanceColor.needsUpdate = true;

  // Salidas
  const top = r.probs.indexOf(Math.max(...r.probs));
  salidas.forEach((s, k) => {
    const p = r.probs[k];
    s.mesh.scale.y = Math.max(0.05, p * 5);
    s.mat.color.set(k === top ? 0xe4572e : 0x2a9d8f).multiplyScalar(0.25 + 0.75 * Math.min(1, p * 2 + (k === top ? 0.5 : 0)));
    s.pct.position.set(CAPAS[6].x, -2.2 + Math.max(0.05, p * 5) + 0.35, posOut[k].z);
    const gx = s.cv.getContext('2d');
    gx.clearRect(0, 0, 512, 128);
    gx.font = 'bold 56px ui-monospace, Consolas, monospace';
    gx.fillStyle = k === top ? '#ffb199' : '#8aa0bd';
    gx.textAlign = 'center'; gx.textBaseline = 'middle';
    gx.fillText(`${(p * 100).toFixed(p > 0.1 ? 0 : 1)}%`, 256, 64);
    s.pct.material.map.needsUpdate = true;
  });

  // Líneas
  conexiones.forEach((cn) => {
    const ia = intensidades(cn.idA, r), ib = intensidades(cn.idB, r);
    const col = cn.attr.array;
    cn.pares.forEach(([i, j], n) => {
      const k = Math.pow(ia[i] * ib[j], 0.5);
      const v = 0.05 + 0.95 * k;
      const o = n * 6;
      col[o] = 0.0 + 0.25 * v; col[o + 1] = 0.55 * v; col[o + 2] = 0.75 * v;
      col[o + 3] = 0.1 * v + 0.0; col[o + 4] = 0.6 * v; col[o + 5] = 0.85 * v;
    });
    cn.attr.needsUpdate = true;
  });

  panel(r, top);
  return r;
}

// ---------- Animación de "recorrido" por las capas ----------
let animInicio = null;
function reproducir() {
  animInicio = performance.now();
  if (rfRes) { rfInicio = animInicio; brilloRF(0, 0); aplicarRF(0); }
  revelado = [0, 0, 0, 0, 0, 0, 0];
}
const PASO = 450; // ms entre capas
function avanzarAnimacion(ahora) {
  if (animInicio !== null) {
    let todas = true;
    for (let k = 0; k < 7; k++) {
      const t = (ahora - animInicio - k * PASO) / 500;
      revelado[k] = Math.min(1, Math.max(0, t));
      if (revelado[k] < 1) todas = false;
    }
    if (todas) animInicio = null;
    const activa = revelado.findIndex((v) => v < 1);
    marcarCapa(animInicio === null ? -1 : activa);
  }
  const ids = ['in', 'c1', 'p1', 'c2', 'p2', 'd1', 'out'];
  ids.forEach((id, k) => {
    const f = 0.08 + 0.92 * revelado[k];
    if (mapas[id]) mapas[id].forEach((p) => p.mesh.material.color.setScalar(f));
  });
  neuronas.material.color.setScalar(0.08 + 0.92 * revelado[5]);
  salidas.forEach((s) => { s.mesh.visible = revelado[6] > 0.02; s.pct.visible = revelado[6] > 0.5; });
  conexiones.forEach((cn) => { cn.mat.opacity = 0.55 * revelado[cn.capaDestinoIdx]; });
}

// ---------- Panel lateral ----------
const elFlujo = document.getElementById('flujo');
const elBarras = document.getElementById('barras');
const elMate = document.getElementById('mate');
const elRes = document.getElementById('resultado');
CAPAS.forEach((c, k) => {
  const d = document.createElement('div');
  d.className = 'capa'; d.id = 'capa-' + k;
  d.innerHTML = `<b>${c.nombre}</b><span>${c.forma}</span><p>${c.txt}</p>`;
  elFlujo.appendChild(d);
});
function marcarCapa(k) {
  CAPAS.forEach((_, i) => document.getElementById('capa-' + i).classList.toggle('activa', i === k));
}
CLASES.forEach((n, k) => {
  const f = document.createElement('div');
  f.className = 'barra'; f.id = 'barra-' + k;
  f.innerHTML = `<span>${n}</span><div class="pistas"><i class="bc"><u></u></i><i class="br"><u></u></i></div><em><b class="c">0%</b><b class="r">0%</b></em>`;
  elBarras.appendChild(f);
});

function panel(r, top) {
  r.probs.forEach((p, k) => {
    const f = document.getElementById('barra-' + k);
    f.querySelector('.bc u').style.width = (p * 100).toFixed(1) + '%';
    f.querySelector('em .c').textContent = pc(p) + '%';
    f.classList.toggle('tc', k === top);
  });
  const mx = Math.max(...r.z);
  const sumaE = r.z.reduce((s, v) => s + Math.exp(v - mx), 0);
  const orden = r.probs.map((p, k) => [p, k]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  elMate.innerHTML =
    `<div>softmax(z<sub>k</sub>) = e<sup>z<sub>k</sub></sup> / Σ e<sup>z<sub>j</sub></sup></div>` +
    orden.map(([p, k]) => `<div>${CLASES[k]}: z = ${r.z[k].toFixed(2)} → e<sup>z</sup>/Σ = ${(Math.exp(r.z[k] - mx)).toFixed(2)}/${sumaE.toFixed(2)} = <b>${pc(p)}%</b></div>`).join('');
  elRes.innerHTML = `<b class="et">CNN</b> ${CLASES[top]} con ${pc(r.probs[top])}% de seguridad`;
}

// ---------- Lienzo para dibujar ----------
const lienzo = document.getElementById('lienzo');
const g = lienzo.getContext('2d', { willReadFrequently: true });
const chico = document.createElement('canvas');
chico.width = chico.height = 28;
const gc = chico.getContext('2d', { willReadFrequently: true });
function limpiarLienzo() { g.fillStyle = '#000'; g.fillRect(0, 0, 224, 224); }
limpiarLienzo();

function desdeLienzo() {
  gc.imageSmoothingEnabled = true; gc.imageSmoothingQuality = 'high';
  gc.drawImage(lienzo, 0, 0, 28, 28);
  const d = gc.getImageData(0, 0, 28, 28).data;
  const img = new Float32Array(784);
  for (let i = 0; i < 784; i++) img[i] = d[i * 4] / 255;
  return img;
}

let dibujando = false, previo = null, pendiente = false;
function posicion(e) {
  const r = lienzo.getBoundingClientRect();
  return [(e.clientX - r.left) * 224 / r.width, (e.clientY - r.top) * 224 / r.height];
}
function trazo(e) {
  const p = posicion(e);
  g.strokeStyle = '#fff'; g.lineWidth = 17; g.lineCap = 'round'; g.lineJoin = 'round';
  g.beginPath(); g.moveTo(...(previo || p)); g.lineTo(...p); g.stroke();
  previo = p;
  if (!pendiente) {
    pendiente = true;
    requestAnimationFrame(() => { pendiente = false; todo(desdeLienzo()); });
  }
}
lienzo.addEventListener('pointerdown', (e) => {
  dibujando = true; previo = null; lienzo.setPointerCapture(e.pointerId);
  if (animInicio === null && revelado.some((v) => v < 1)) revelado = [1, 1, 1, 1, 1, 1, 1];
  cortarAnimRF();
  claseReal = null;
  trazo(e);
});
lienzo.addEventListener('pointermove', (e) => { if (dibujando) trazo(e); });
lienzo.addEventListener('pointerup', () => { dibujando = false; previo = null; });
document.getElementById('limpiar').onclick = () => { limpiarLienzo(); claseReal = null; todo(new Float32Array(784)); };

// ---------- Ejemplos reales del conjunto de prueba ----------
let claseReal = null; // la prenda verdadera, solo si la imagen viene del dataset
function cargarEjemplo(ej) {
  claseReal = ej.clase;
  const img = new Float32Array(784);
  const tmp = new ImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    img[i] = ej.pix[i] / 255;
    tmp.data[i * 4] = tmp.data[i * 4 + 1] = tmp.data[i * 4 + 2] = ej.pix[i];
    tmp.data[i * 4 + 3] = 255;
  }
  gc.putImageData(tmp, 0, 0);
  limpiarLienzo();
  g.imageSmoothingEnabled = false;
  g.drawImage(chico, 0, 0, 224, 224);
  g.imageSmoothingEnabled = true;
  todo(img);
  reproducir();
}
const elEj = document.getElementById('ejemplos');
const rotacion = new Array(10).fill(0);
CLASES.forEach((n, k) => {
  const b = document.createElement('button');
  b.textContent = n;
  b.onclick = () => {
    const lista = DATOS.ejemplos.filter((e) => e.clase === k);
    cargarEjemplo(lista[rotacion[k]++ % lista.length]);
  };
  elEj.appendChild(b);
});
(DATOS.casos || []).forEach((c) => {
  const b = document.createElement('button');
  b.textContent = c.nombre;
  b.title = `Imagen ${c.idx} del conjunto de prueba (real: ${CLASES[c.clase]})`;
  b.onclick = () => cargarEjemplo(c);
  document.getElementById('casos').appendChild(b);
});
document.getElementById('replay').onclick = reproducir;
const bRot = document.getElementById('rotar');
bRot.onclick = () => { orbit.autoRotate = !orbit.autoRotate; bRot.classList.toggle('on', orbit.autoRotate); };
document.getElementById('vista').onclick = () => irAVista();

// ---------- Etiqueta al pasar el mouse por un mapa ----------
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tip = document.getElementById('tip');
renderer.domElement.addEventListener('pointermove', (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, cam);
  const hit = ray.intersectObjects(todosLosPlanos, false).find((h) => visibleDeVerdad(h.object));
  const hitA = rfG.visible && bosque ? ray.intersectObject(arboles, false)[0] : null;
  if (hitA && (!hit || hitA.distance < hit.distance)) {
    tip.style.display = 'block'; tip.style.left = e.clientX + 14 + 'px'; tip.style.top = e.clientY + 14 + 'px';
    tip.textContent = `Árbol ${hitA.instanceId + 1} · vota ${CLASES[rfRes.porArbol[hitA.instanceId]]} (clic para ver sus preguntas)`;
  } else if (hit) { tip.style.display = 'block'; tip.style.left = e.clientX + 14 + 'px'; tip.style.top = e.clientY + 14 + 'px'; tip.textContent = hit.object.userData.nombre; }
  else tip.style.display = 'none';
});

// =====================================================================================
// ---------- Random Forest: 200 árboles que votan ----------
// =====================================================================================
const COL_CLASE = CLASES.map((_, k) => new THREE.Color().setHSL(k / 10, 0.72, 0.58));
const hexClase = (k) => '#' + COL_CLASE[k].getHexString();
const OSCURO = new THREE.Color(0x0a1220);
const NA = DATOS.rf.arboles;
const RX = { in: 0, imp: 7.5, arb: 18.5, out: 31 }; // posición x de cada etapa
const ETAPAS_RF = [
  { nombre: 'Entrada', forma: '784 píxeles', x: RX.in, txt: 'La imagen aplanada: el bosque recibe 784 números sueltos y no sabe qué píxel está al lado de cuál.' },
  { nombre: 'Importancia', forma: '28×28', x: RX.imp, txt: 'Qué píxeles le sirvieron más al bosque al entrenar (más claro = más importante). Es fijo: no depende de la imagen.' },
  { nombre: `${NA} árboles`, forma: '1,8 M nodos', x: RX.arb, txt: 'Cada cubo es un árbol. Hace preguntas sí/no sobre píxeles sueltos hasta una hoja y vota una prenda (el color). Clic en un cubo para ver sus preguntas.' },
  { nombre: 'Votación', forma: '10', x: RX.out, txt: 'Gana la prenda más votada. La proporción de votos es la probabilidad: no hay softmax.' },
];

const rfIn = crearPlano(2.8, 28, 'Entrada (en naranja: los píxeles que pregunta el árbol elegido)');
rfIn.mesh.position.x = RX.in; rfG.add(rfIn.mesh);
const rfImp = crearPlano(3.4, 28, 'Importancia de cada píxel (feature_importances_)');
rfImp.mesh.position.x = RX.imp; rfG.add(rfImp.mesh);
{
  const imp = b64ABytes(DATOS.rf.importancia);
  for (let i = 0; i < 784; i++) {
    const [r, g2, b] = color(Math.pow(imp[i] / 255, 0.6));
    rfImp.data.set([r, g2, b, 255], i * 4);
  }
  rfImp.tex.needsUpdate = true;
}

// Los árboles: una grilla de cubos
const COLS_A = 20, SEP_A = 0.6;
const arboles = new THREE.InstancedMesh(new THREE.BoxGeometry(0.4, 0.4, 0.4), new THREE.MeshBasicMaterial({ color: 0xffffff }), NA);
const posArbol = [];
{
  const m = new THREE.Matrix4();
  const filas = Math.ceil(NA / COLS_A);
  for (let i = 0; i < NA; i++) {
    const c = i % COLS_A, f = Math.floor(i / COLS_A);
    const p = new THREE.Vector3(RX.arb, ((filas - 1) / 2 - f) * SEP_A, (c - (COLS_A - 1) / 2) * SEP_A);
    posArbol.push(p);
    m.setPosition(p);
    arboles.setMatrixAt(i, m);
    arboles.setColorAt(i, OSCURO);
  }
  rfG.add(arboles);
}
const marcador = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.62, 0.62, 0.62)), new THREE.LineBasicMaterial({ color: 0xffffff }));
rfG.add(marcador);

// Columnas de votos
const salidasRF = [];
const zOut = (k) => (k - 4.5) * 0.95;
for (let k = 0; k < 10; k++) {
  const geo = new THREE.BoxGeometry(0.5, 1, 0.5);
  geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshBasicMaterial({ color: COL_CLASE[k].clone() });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(RX.out, -2.2, zOut(k));
  mesh.scale.y = 0.05;
  rfG.add(mesh);
  const et = etiqueta(CLASES[k], 56, hexClase(k));
  et.scale.set(2.2, 0.55, 1);
  et.position.set(RX.out, -2.75, zOut(k));
  rfG.add(et);
  const pct = etiqueta('0%', 56, '#ffffff');
  pct.scale.set(1.4, 0.35, 1);
  rfG.add(pct);
  salidasRF.push({ mesh, mat, pct, cv: pct.material.map.image });
}
ETAPAS_RF.forEach((c) => {
  const s = etiqueta(c.nombre, 54, '#ffcf99');
  s.scale.set(4.2, 1.05, 1); s.position.set(c.x, altoCartel, 0); rfG.add(s);
  const s2 = etiqueta(c.forma, 48, '#5b6b86');
  s2.scale.set(3.4, 0.85, 1); s2.position.set(c.x, altoCartel - 0.65, 0); rfG.add(s2);
});

// Una línea de cada árbol a la prenda que votó
const lPos = new Float32Array(NA * 6), lCol = new Float32Array(NA * 6);
const geoVotos = new THREE.BufferGeometry();
geoVotos.setAttribute('position', new THREE.BufferAttribute(lPos, 3));
geoVotos.setAttribute('color', new THREE.BufferAttribute(lCol, 3));
const lineasVotos = new THREE.LineSegments(geoVotos, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false }));
rfG.add(lineasVotos);
// Líneas de la entrada a algunos árboles (decorativas: todos los árboles miran la misma imagen)
{
  const n = 60, pos = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    const a = posArbol[Math.floor((i * NA) / n)];
    pos.set([RX.in + 0.02, (Math.random() - 0.5) * 2.4, (Math.random() - 0.5) * 2.4, a.x - 0.2, a.y, a.z], i * 6);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  rfG.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x1d4a5a, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false })));
}

// ---------- Estado y actualización del bosque ----------
let bosque = null;        // se arma cuando termina de descomprimirse
let rfRes = null, rfPix = null, rfCam = null, arbolSel = 0, rfN = NA, rfInicio = null;

function pintarEntradaRF() {
  for (let i = 0; i < 784; i++) {
    const [r, g2, b] = color(Math.pow(rfPix[i] / 255, 0.8));
    rfIn.data.set([r, g2, b, 255], i * 4);
  }
  if (rfCam) for (const p of rfCam.pasos) rfIn.data.set([255, 140, 40, 255], p.pixel * 4);
  rfIn.tex.needsUpdate = true;
}

// Muestra los votos de los primeros n árboles (n < NA durante la animación)
function aplicarRF(n) {
  rfN = n;
  const por = rfRes.porArbol;
  const cnt = new Array(10).fill(0);
  for (let i = 0; i < NA; i++) {
    const ya = i < n;
    if (ya) cnt[por[i]]++;
    const c = ya ? COL_CLASE[por[i]] : OSCURO;
    arboles.setColorAt(i, c);
    const v = ya ? 0.8 : 0;
    lCol.set([c.r * v, c.g * v, c.b * v, c.r * v, c.g * v, c.b * v], i * 6);
  }
  arboles.instanceColor.needsUpdate = true;
  geoVotos.attributes.color.needsUpdate = true;
  const lider = cnt.indexOf(Math.max(...cnt));
  salidasRF.forEach((s, k) => {
    const p = cnt[k] / NA;
    const alto = Math.max(0.05, p * 5);
    s.mesh.scale.y = alto;
    s.mat.color.copy(COL_CLASE[k]).multiplyScalar(k === lider && n > 0 ? 1 : 0.45);
    s.pct.position.set(RX.out, -2.2 + alto + 0.35, zOut(k));
    s.pct.visible = n > 0;
    const gx = s.cv.getContext('2d');
    gx.clearRect(0, 0, 512, 128);
    gx.font = 'bold 56px ui-monospace, Consolas, monospace';
    gx.fillStyle = k === lider ? '#ffffff' : '#8aa0bd';
    gx.textAlign = 'center'; gx.textBaseline = 'middle';
    gx.fillText(`${pc(p, p > 0.1 || p === 0 ? 0 : 1)}%`, 256, 64);
    s.pct.material.map.needsUpdate = true;
  });
}

function seleccionarArbol(t) {
  arbolSel = t;
  marcador.position.copy(posArbol[t]);
  if (!rfRes) return;
  rfCam = camino(bosque, t, rfPix);
  pintarEntradaRF();
  panelArbol();
}

function actualizarRF(img) {
  rfPix = Uint8Array.from(img, (v) => Math.round(v * 255));
  rfRes = votar(bosque, rfPix);
  for (let i = 0; i < NA; i++) {
    const a = posArbol[i];
    lPos.set([a.x + 0.22, a.y, a.z, RX.out - 0.3, -1.6, zOut(rfRes.porArbol[i])], i * 6);
  }
  geoVotos.attributes.position.needsUpdate = true;
  seleccionarArbol(arbolSel);
  aplicarRF(rfInicio === null ? NA : rfN);
  panelRF();
}

let imgActual = new Float32Array(784);
function todo(img) {
  imgActual = img;
  actualizar(img);
  if (bosque) actualizarRF(img);
  panelComp();
}

// Animación: entrada -> importancia -> los árboles votan de a uno -> columnas
const lim = (v) => Math.min(1, Math.max(0, v));
function brilloRF(a, b) { rfIn.mesh.material.color.setScalar(0.08 + 0.92 * a); rfImp.mesh.material.color.setScalar(0.08 + 0.92 * b); }
function animarRF(ahora) {
  if (rfInicio === null || !rfRes) return;
  const e = ahora - rfInicio;
  const f = lim((e - 900) / 1800);
  brilloRF(lim(e / 350), lim((e - 350) / 350));
  const n = Math.round(f * NA);
  if (n !== rfN) aplicarRF(n);
  marcarEtapaRF(e < 350 ? 0 : e < 900 ? 1 : f < 1 ? 2 : 3);
  if (e > 3100) { rfInicio = null; marcarEtapaRF(-1); }
}
function cortarAnimRF() {
  if (rfInicio === null) return;
  rfInicio = null; brilloRF(1, 1); marcarEtapaRF(-1);
  if (rfRes) aplicarRF(NA);
}

// ---------- Paneles del bosque y de la comparación ----------
const elFlujoRF = document.getElementById('flujo-rf');
ETAPAS_RF.forEach((c, k) => {
  const d = document.createElement('div');
  d.className = 'capa'; d.id = 'etapa-' + k;
  d.innerHTML = `<b>${c.nombre}</b><span>${c.forma}</span><p>${c.txt}</p>`;
  elFlujoRF.appendChild(d);
});
function marcarEtapaRF(k) { ETAPAS_RF.forEach((_, i) => document.getElementById('etapa-' + i).classList.toggle('activa', i === k)); }

const topDe = (a) => a.indexOf(Math.max(...a));
function panelRF() {
  const top = topDe(rfRes.probs);
  rfRes.probs.forEach((p, k) => {
    const f = document.getElementById('barra-' + k);
    const u = f.querySelector('.br u');
    u.style.width = (p * 100).toFixed(1) + '%';
    u.style.background = hexClase(k);
    u.style.opacity = k === top ? 1 : 0.55;
    f.querySelector('em .r').textContent = pc(p) + '%';
    f.classList.toggle('tr', k === top);
  });
  const orden = rfRes.probs.map((p, k) => [p, k]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  document.getElementById('mate-rf').innerHTML =
    `<div>probabilidad = votos de la prenda / ${NA} árboles</div>` +
    orden.filter(([p]) => p > 0).map(([p, k]) => `<div><span class="pto" style="background:${hexClase(k)}"></span>${CLASES[k]}: ${rfRes.votos[k]} de ${NA} = <b>${pc(p)}%</b></div>`).join('');
  document.getElementById('resultado-rf').innerHTML = `<b class="et">Random Forest</b> ${CLASES[top]} con ${pc(rfRes.probs[top])}% de los votos`;
}

function panelArbol() {
  const v = rfCam.clase;
  const filas = rfCam.pasos.map((p) => {
    const f = Math.floor(p.pixel / 28), c = p.pixel % 28;
    const u = ((p.umbral + 0.5) / 255).toFixed(2).replace('.', ',');
    return `<li>¿píxel (fila ${f}, col ${c}) &gt; ${u}? <span>vale ${(p.valor / 255).toFixed(2).replace('.', ',')} → ${p.izq ? 'No' : 'Sí'}</span></li>`;
  }).join('');
  document.getElementById('arbol').innerHTML =
    `<div class="cab">Árbol <b>${arbolSel + 1}</b> de ${NA} · vota <b style="color:${hexClase(v)}">${CLASES[v]}</b> después de ${rfCam.pasos.length} preguntas` +
    `<div class="fila"><button id="ant">◀</button><button id="sig">▶</button><button id="azar">Otro al azar</button></div></div>` +
    `<ol class="preguntas">${filas}<li class="hoja">Hoja → <b style="color:${hexClase(v)}">${CLASES[v]}</b></li></ol>` +
    `<p class="nota">En naranja, sobre la entrada, los píxeles que pregunta este árbol: mira píxeles sueltos, no formas.</p>`;
  document.getElementById('ant').onclick = () => seleccionarArbol((arbolSel + NA - 1) % NA);
  document.getElementById('sig').onclick = () => seleccionarArbol((arbolSel + 1) % NA);
  document.getElementById('azar').onclick = () => seleccionarArbol(Math.floor(Math.random() * NA));
}

function panelComp() {
  const el = document.getElementById('veredicto');
  if (!ultimo || !rfRes) { el.innerHTML = ''; return; }
  const tc = topDe(ultimo.probs), tr = topDe(rfRes.probs);
  let txt = tc === tr
    ? `<b class="ok">Coinciden:</b> los dos dicen ${CLASES[tc]}.`
    : `<b class="no">No coinciden:</b> la CNN dice ${CLASES[tc]} y el bosque ${CLASES[tr]}.`;
  if (claseReal !== null) {
    const marca = (k) => (k === claseReal ? '<b class="ok">acierta</b>' : '<b class="no">falla</b>');
    txt += `<br>Real: <b>${CLASES[claseReal]}</b> · CNN ${marca(tc)} · bosque ${marca(tr)}`;
  }
  el.innerHTML = txt;
}

// ---------- Modos: CNN, Random Forest o los dos ----------
let modo = 'cnn';
const VISTAS = {
  cnn: VISTA,
  rf: { pos: [-15, 9, 31], mira: [16, -0.5, 0] },
  comp: { pos: [-18, 2, 42], mira: [16, -6.5, 0] },
};
function irAVista() { const v = VISTAS[modo]; cam.position.set(...v.pos); orbit.target.set(...v.mira); }
const exCNN = pc(DATOS.exactitud, 2) + '%', exRF = pc(DATOS.rf.exactitud, 2) + '%';
function setModo(m) {
  modo = m;
  document.body.dataset.modo = m;
  cnnG.visible = m !== 'rf';
  rfG.visible = m !== 'cnn';
  rfG.position.y = m === 'comp' ? -13 : 0;
  document.querySelectorAll('[data-en]').forEach((el) => { el.hidden = !el.dataset.en.split(' ').includes(m); });
  document.querySelectorAll('#modos button').forEach((b) => b.classList.toggle('on', b.dataset.modo === m));
  document.getElementById('exactitud').textContent = m === 'cnn' ? `CNN ${exCNN}` : m === 'rf' ? `Random Forest ${exRF}` : `CNN ${exCNN} · Random Forest ${exRF}`;
  irAVista();
}
document.querySelectorAll('#modos button').forEach((b) => { b.onclick = () => setModo(b.dataset.modo); });

// Clic en un árbol (sin arrastrar) para elegirlo
let bajo = null;
renderer.domElement.addEventListener('pointerdown', (e) => { bajo = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!bajo || Math.hypot(e.clientX - bajo[0], e.clientY - bajo[1]) > 4 || !rfG.visible || !bosque) return;
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, cam);
  const hit = ray.intersectObject(arboles, false)[0];
  if (hit) seleccionarArbol(hit.instanceId);
});
function visibleDeVerdad(o) { for (; o; o = o.parent) if (!o.visible) return false; return true; }

// El bosque viene comprimido con gzip: se descomprime con la API del navegador
async function cargarBosque() {
  const flujo = new Blob([b64ABytes(DATOS.rf.datos)]).stream().pipeThrough(new DecompressionStream('gzip'));
  const crudo = new Uint8Array(await new Response(flujo).arrayBuffer());
  bosque = construirBosque(crudo, DATOS.rf);
  document.querySelectorAll('#modos button').forEach((b) => { b.disabled = false; });
  document.getElementById('cargando').hidden = true;
  todo(imgActual);
}

// ---------- Arranque ----------
ajustar();
function bucle(t) {
  requestAnimationFrame(bucle);
  avanzarAnimacion(t);
  animarRF(t);
  orbit.update();
  renderer.render(escena, cam);
}
requestAnimationFrame(bucle);
setModo('cnn');
cargarEjemplo(DATOS.ejemplos.find((e) => e.clase === 9));
cargarBosque().catch((err) => { document.getElementById('cargando').textContent = 'No se pudo cargar el bosque: ' + err.message; });
window.__viz = { todo, setModo, adelante: (img) => adelante(modelo, img), get ultimo() { return ultimo; }, get rf() { return rfRes; }, get bosque() { return bosque; } };
