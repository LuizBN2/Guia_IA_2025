import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.tokenize import sent_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 5000
MAXLEN = 40

def cargar_modelo_y_tokenizer(tipo_modelo):
    modelo = load_model(f"modelos/modelo_{tipo_modelo.lower()}.keras")
    with open("modelos/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return modelo, tokenizer

def tokenizar_oraciones(oraciones, tokenizer):
    secuencias = tokenizer.texts_to_sequences(oraciones)
    secuencias = pad_sequences(secuencias, maxlen=MAXLEN, padding='post', truncating='post')
    return secuencias

def generar_texto_desde_corpus(corpus_path, modelo, tokenizer, umbral=0.5):
    with open(corpus_path, encoding='utf-8') as f:
        texto = f.read()
    oraciones = sent_tokenize(texto)

    secuencias = tokenizar_oraciones(oraciones, tokenizer)
    predicciones = modelo.predict(secuencias)

    oraciones_generadas = [
        oracion for oracion, prob in zip(oraciones, predicciones)
        if prob >= umbral
    ]

    return " ".join(oraciones_generadas), oraciones_generadas, predicciones
