# Visualizador CNN vs Random Forest · FashionMNIST

Página interactiva en three.js que muestra por dentro los dos modelos del TP4 (Introducción al Aprendizaje Automático): una CNN en Keras y un Random Forest de scikit-learn, entrenados con las mismas 54.000 imágenes de FashionMNIST.

**Para usarlo:** abrir `visualizador/visualizador_cnn.html` con doble clic. Funciona sin servidor ni internet: los pesos de la CNN y los 200 árboles del bosque van dentro del HTML.

- Modos **CNN**, **Random Forest** y **Comparar**.
- Dibujar una prenda o cargar ejemplos reales del conjunto de prueba, incluidos los dos casos de la presentación.
- CNN: mapas de activación de cada capa y probabilidades con softmax.
- Random Forest: importancia de cada píxel, el voto de cada árbol y las preguntas que hace un árbol elegido.

Exactitud en prueba: CNN 92,71 % · Random Forest 87,58 %.

## Regenerar

Desde esta carpeta:

```bash
python entrenar_y_exportar.py   # entrena la CNN y escribe visualizador/datos.json
python entrenar_rf.py           # entrena el Random Forest (modelo_rf.joblib)
python exportar_rf.py           # agrega el bosque a visualizador/datos.json
cd visualizador/src
node verificar.mjs              # CNN en JS vs Keras
node verificar_rf.mjs           # Random Forest en JS vs scikit-learn
NODE_PATH=<node_modules con three y esbuild> node construir.js
```
