// Junta plantilla + código (con three.js) + pesos en un único HTML autónomo.
// Uso:  NODE_PATH=<carpeta con node_modules de three y esbuild> node construir.js
// Antes: python entrenar_y_exportar.py (CNN) y python exportar_rf.py (Random Forest), desde la carpeta del TP.
const fs = require('fs');
const path = require('path');
const esbuild = require('esbuild');

const aqui = __dirname;
const salida = path.join(aqui, '..', 'visualizador_cnn.html');

const r = esbuild.buildSync({
  entryPoints: [path.join(aqui, 'main.js')],
  bundle: true, minify: true, format: 'iife', write: false, target: 'es2020',
  nodePaths: (process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean),
});
const js = r.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const datos = fs.readFileSync(path.join(aqui, '..', 'datos.json'), 'utf8');
const html = fs.readFileSync(path.join(aqui, 'plantilla.html'), 'utf8')
  .replace('/*DATOS*/', () => datos)
  .replace('/*BUNDLE*/', () => js);
fs.writeFileSync(salida, html);
console.log('HTML:', (html.length / 1e6).toFixed(2), 'MB ->', salida);
