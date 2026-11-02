# Inside the Machine · CNN vs Random Forest en FashionMNIST

Página interactiva en three.js que muestra por dentro los dos modelos del TP4 (Introducción al Aprendizaje Automático): una CNN en Keras y un Random Forest de scikit-learn, entrenados con las mismas 54.000 imágenes de FashionMNIST.

**Para usarlo:** abrir `visualizador/visualizador_cnn.html` con doble clic. Funciona sin servidor ni internet: los pesos de la CNN, los 200 árboles del bosque y las fuentes van dentro del HTML.

- Modos **CNN**, **Random Forest** y **Comparar** (las dos arquitecturas a la vez, una sobre la otra).
- Reproducir, pausar, avanzar paso a paso, reiniciar y cambiar la velocidad de la secuencia. Barra espaciadora y flecha derecha también funcionan.
- Entrada: dibujar a mano, elegir una prenda, una muestra aleatoria entre 500 imágenes de prueba, cargar una imagen propia o los dos casos de la presentación.
- CNN: el kernel 3×3 recorre la entrada mientras se llenan los mapas, con la cuenta Σ x·w + b real del filtro más activo; el pooling se ve reduciendo los mapas a la mitad; conexiones con opacidad según la activación.
- Random Forest: cada árbol crece siguiendo su recorrido real para la imagen (más las ramas descartadas), los votos convergen en el conteo y se puede elegir un árbol para ver sus preguntas y los píxeles que mira.
- Los paneles laterales se pueden plegar. En tablet y celular se apilan.

Exactitud en prueba: CNN 92,71 % · Random Forest 87,58 %.

La CNN reporta la salida softmax (no calibrada) y el bosque la proporción de votos de sus árboles: la página no las presenta como equivalentes.

## Regenerar

Desde esta carpeta:

```bash
python entrenar_y_exportar.py   # entrena la CNN y escribe visualizador/datos.json
python entrenar_rf.py           # entrena el Random Forest (modelo_rf.joblib)
python exportar_rf.py           # agrega el bosque, los casos y 500 muestras a visualizador/datos.json
cd visualizador/src
node verificar.mjs              # CNN en JS vs Keras
node verificar_rf.mjs           # Random Forest en JS vs scikit-learn
NODE_PATH=<node_modules con three, esbuild, @fontsource-variable/inter y @fontsource-variable/jetbrains-mono> node construir.js
```

Código en `visualizador/src`: `cnn.js` y `rf.js` son los modelos (sin cambios visuales), `vista_cnn.js` y `vista_rf.js` las escenas 3D, `comun.js` las piezas compartidas y `main.js` la interfaz y el reloj de la secuencia.
