import nltk
nltk.download('punkt')
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from pages.entrenamiento_modelo_25 import limpiar_y_filtrar


def resumir_noticia(noticia, modelo, tokenizer, umbral, max_len=50):
    oraciones = [s.strip() for s in nltk.sent_tokenize(noticia, language='spanish') if len(s.strip()) > 0]
    X_filtrado = [limpiar_y_filtrar(oracion) for oracion in oraciones]
    #tokenizer.fit_on_texts(X_filtrado) para wordclud
    secuencias = tokenizer.texts_to_sequences(oraciones)
    padded = pad_sequences(secuencias, maxlen=max_len, padding='post', truncating='post')
    predicciones = modelo.predict(padded)
    resumen = [oraciones[i] for i in range(len(oraciones)) if predicciones[i] >= umbral]
    return resumen, X_filtrado
