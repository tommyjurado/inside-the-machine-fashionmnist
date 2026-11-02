// Compara el Random Forest escrito en JavaScript con las probabilidades que dio scikit-learn.
import fs from 'fs';
import zlib from 'zlib';
import { b64ABytes, construirBosque, votar } from './rf.js';

const datos = JSON.parse(fs.readFileSync(new URL('../datos.json', import.meta.url)));
const crudo = new Uint8Array(zlib.gunzipSync(b64ABytes(datos.rf.datos)));
const bosque = construirBosque(crudo, datos.rf);
let peor = 0, aciertos = 0, mismaClase = 0;
for (const ej of datos.ejemplos) {
  const r = votar(bosque, Uint8Array.from(ej.pix));
  r.probs.forEach((p, k) => { peor = Math.max(peor, Math.abs(p - ej.rf[k])); });
  const top = (a) => a.indexOf(Math.max(...a));
  if (top(r.probs) === ej.clase) aciertos++;
  if (top(r.probs) === top(ej.rf)) mismaClase++;
}
console.log(`ejemplos: ${datos.ejemplos.length} | diferencia máxima JS vs scikit-learn: ${peor.toExponential(2)} | aciertos: ${aciertos} | misma clase ganadora que scikit-learn: ${mismaClase}/${datos.ejemplos.length}`);
// scikit-learn promedia la probabilidad de cada hoja; aca cada arbol da un voto. Difieren como mucho por 1 arbol (0,005).
if (peor > 0.0101 || mismaClase < datos.ejemplos.length) { console.error('NO COINCIDE'); process.exit(1); }
