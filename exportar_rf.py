"""Exporta el Random Forest (modelo_rf.joblib) a un formato compacto para el visualizador.

Agrega la clave "rf" a visualizador/datos.json. Por nodo (en el orden de scikit-learn, preorden:
el hijo izquierdo siempre es el nodo siguiente):
  feat  uint16  pixel que se pregunta (65535 = hoja)
  thr   uint8   umbral entero 0-255 ("pixel <= thr" va a la izquierda); en hojas, la clase
  der   uint16  salto hasta el hijo derecho (relativo al nodo; 0 en hojas)
Todo concatenado por arbol y comprimido con gzip + base64.
"""
import base64, gzip, json
import joblib, keras, numpy as np

rf = joblib.load('modelo_rf.joblib')
feat, thr, der, inicio = [], [], [], [0]
for e in rf.estimators_:
    t = e.tree_
    izq, dr, f, th = t.children_left, t.children_right, t.feature, t.threshold
    n = t.node_count
    assert all(izq[i] == i + 1 for i in range(n) if izq[i] != -1), 'no es preorden'
    clase = e.classes_[t.value[:, 0, :].argmax(1)]
    for i in range(n):
        if izq[i] == -1:
            feat.append(65535); thr.append(int(clase[i])); der.append(0)
        else:
            u = int(np.floor(th[i] * 255 + 1e-3))
            assert 0 <= u <= 254
            feat.append(int(f[i])); thr.append(u)
            salto = int(dr[i] - i); assert salto < 65536
            der.append(salto)
    inicio.append(len(feat))

b = (np.array(feat, '<u2').tobytes(), np.array(thr, 'u1').tobytes(), np.array(der, '<u2').tobytes())
crudo = b[0] + b[1] + b[2]
gz = gzip.compress(crudo, 9)
print(f'nodos: {len(feat):,} | crudo {len(crudo)/1e6:.1f} MB | gzip {len(gz)/1e6:.1f} MB | b64 {len(gz)*4/3/1e6:.1f} MB')

imp = rf.feature_importances_
imp8 = np.round(imp / imp.max() * 255).astype('u1')
(xt, yt), (xs, ys) = keras.datasets.fashion_mnist.load_data()

datos = json.load(open('visualizador/datos.json'))
datos['rf'] = {
    'arboles': len(rf.estimators_), 'nodos': len(feat), 'inicio': inicio,
    'datos': base64.b64encode(gz).decode(),
    'importancia': base64.b64encode(imp8.tobytes()).decode(),
    'exactitud': round(float((rf.predict((xs / 255.).astype('float32').reshape(len(xs), -1)) == ys).mean()), 4),
}
# probabilidades de scikit-learn para los mismos ejemplos, para verificar el JS
X = np.array([e['pix'] for e in datos['ejemplos']], 'float32') / 255.
for e, p in zip(datos['ejemplos'], rf.predict_proba(X)):
    e['rf'] = [round(float(v), 6) for v in p]
# los dos casos de la demostración del PPT (slide 17): índice 0 y el primer error de la CNN (índice 23)
datos['casos'] = [{'nombre': nombre, 'idx': i, 'clase': int(ys[i]), 'pix': xs[i].flatten().astype(int).tolist()}
                  for nombre, i in [('Caso 1 · acierto', 0), ('Caso 2 · error', 23)]]
# 500 imágenes del conjunto de prueba (50 por prenda) para la muestra aleatoria y el selector de categoría
rng = np.random.default_rng(42)
idx = np.concatenate([rng.choice(np.where(ys == k)[0], 50, replace=False) for k in range(10)])
datos['muestras'] = {'idx': idx.tolist(), 'clase': ys[idx].tolist(),
                     'pix': base64.b64encode(xs[idx].astype('u1').tobytes()).decode()}
json.dump(datos, open('visualizador/datos.json', 'w'))
print('listo')
