// Compara la CNN escrita en JavaScript con las probabilidades que dio Keras para los mismos ejemplos.
import fs from 'fs';
import { construirModelo, adelante } from './cnn.js';

const datos = JSON.parse(fs.readFileSync(new URL('../datos.json', import.meta.url)));
const modelo = construirModelo(datos.pesos);
let peor = 0, aciertos = 0;
for (const ej of datos.ejemplos) {
  const img = Float32Array.from(ej.pix, (v) => v / 255);
  const r = adelante(modelo, img);
  r.probs.forEach((p, k) => { peor = Math.max(peor, Math.abs(p - ej.keras[k])); });
  if (r.probs.indexOf(Math.max(...r.probs)) === ej.clase) aciertos++;
}
console.log(`ejemplos: ${datos.ejemplos.length} | diferencia máxima JS vs Keras: ${peor.toExponential(2)} | aciertos: ${aciertos}`);
if (peor > 1e-3) { console.error('NO COINCIDE'); process.exit(1); }
