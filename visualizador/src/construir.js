// Junta plantilla + código (con three.js) + pesos + fuentes en un único HTML autónomo.
// Uso:  NODE_PATH=<node_modules con three, esbuild, @fontsource-variable/inter y @fontsource-variable/jetbrains-mono> node construir.js
// Antes: python entrenar_y_exportar.py (CNN) y python exportar_rf.py (Random Forest), desde la carpeta del TP.
const fs = require('fs');
const path = require('path');
const esbuild = require('esbuild');

const aqui = __dirname;
const salida = path.join(aqui, '..', 'visualizador_cnn.html');
const rutas = (process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean);

const r = esbuild.buildSync({
  entryPoints: [path.join(aqui, 'main.js')],
  bundle: true, minify: true, format: 'iife', write: false, target: 'es2020',
  nodePaths: rutas,
});
const js = r.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');

// Fuentes embebidas (subconjunto latino): la página funciona sin internet
function fuente(familia, archivo) {
  for (const base of rutas) {
    const f = path.join(base, archivo);
    if (fs.existsSync(f)) {
      const b64 = fs.readFileSync(f).toString('base64');
      return `@font-face{font-family:"${familia}";font-style:normal;font-display:swap;font-weight:100 900;src:url(data:font/woff2;base64,${b64}) format("woff2");}`;
    }
  }
  console.warn('Fuente no encontrada, se usa la del sistema:', archivo);
  return '';
}
const fuentes = fuente('Inter Variable', '@fontsource-variable/inter/files/inter-latin-wght-normal.woff2') +
  fuente('JetBrains Mono Variable', '@fontsource-variable/jetbrains-mono/files/jetbrains-mono-latin-wght-normal.woff2');

const datos = fs.readFileSync(path.join(aqui, '..', 'datos.json'), 'utf8');
const html = fs.readFileSync(path.join(aqui, 'plantilla.html'), 'utf8')
  .replace('/*FUENTES*/', () => fuentes)
  .replace('/*DATOS*/', () => datos)
  .replace('/*BUNDLE*/', () => js);
fs.writeFileSync(salida, html);
console.log('HTML:', (html.length / 1e6).toFixed(2), 'MB ->', salida);
