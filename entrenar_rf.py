"""Entrena el Random Forest del PPT (200 arboles, sqrt, semilla 42, 54.000 imagenes) y lo guarda."""
import time, joblib
import numpy as np
import keras
from sklearn.ensemble import RandomForestClassifier

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# mismas 54.000 imagenes que la CNN: validation_split=0.1 de Keras reserva el ultimo 10%
n = 54000
X = (x_train[:n].reshape(n, -1) / 255.0).astype('float32')
Xt = (x_test.reshape(len(x_test), -1) / 255.0).astype('float32')

rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=42, n_jobs=-1)
t = time.time(); rf.fit(X, y_train[:n]); print(f'entrenamiento: {time.time()-t:.0f} s')
pred = rf.predict(Xt)
print(f'EXACTITUD EN PRUEBA: {(pred == y_test).mean():.4f}  errores: {(pred != y_test).sum()}')
print('nodos totales:', sum(e.tree_.node_count for e in rf.estimators_))
joblib.dump(rf, 'modelo_rf.joblib', compress=3)
