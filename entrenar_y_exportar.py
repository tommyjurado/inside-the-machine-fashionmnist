"""Reentrena la misma CNN del notebook y exporta sus pesos para el visualizador 3D.

Genera:
  - modelo_cnn.keras          (modelo completo)
  - visualizador/datos.json   (pesos en base64 float32 + ejemplos del test + probabilidades de Keras)
"""
import base64, json, os
import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = (x_train / 255.0).astype('float32')[..., np.newaxis]
x_test = (x_test / 255.0).astype('float32')[..., np.newaxis]

modelo = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax'),
])
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
parar = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
modelo.fit(x_train, y_train, validation_split=0.1, epochs=20, batch_size=128, callbacks=[parar], verbose=2)

_, acc = modelo.evaluate(x_test, y_test, verbose=0)
print(f'EXACTITUD EN PRUEBA: {acc:.4f}')
modelo.save('modelo_cnn.keras')

# Pesos en el orden de Keras: conv (alto, ancho, canal_in, canal_out), dense (in, out)
pesos = {}
for nombre, capa in zip(['conv1', 'conv2', 'dense1', 'dense2'],
                        [l for l in modelo.layers if l.weights]):
    w, b = capa.get_weights()
    pesos[nombre] = {
        'w_shape': list(w.shape),
        'w': base64.b64encode(w.astype('<f4').tobytes()).decode(),
        'b': base64.b64encode(b.astype('<f4').tobytes()).decode(),
    }

# 3 ejemplos por clase (los primeros del test) + lo que predice Keras, para verificar el JS
ejemplos = []
for clase in range(10):
    for i in np.where(y_test == clase)[0][:3]:
        p = modelo.predict(x_test[i:i + 1], verbose=0)[0]
        ejemplos.append({'idx': int(i), 'clase': clase,
                         'pix': (x_test[i, :, :, 0] * 255).round().astype(int).flatten().tolist(),
                         'keras': [round(float(v), 6) for v in p]})

os.makedirs('visualizador', exist_ok=True)
with open('visualizador/datos.json', 'w') as f:
    json.dump({'exactitud': round(float(acc), 4), 'pesos': pesos, 'ejemplos': ejemplos}, f)
print('listo')
