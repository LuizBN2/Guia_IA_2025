import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import os

from modelo_factory_secuencial import (
    crear_modelo_rnn_secuencial,
    crear_modelo_cnn_secuencial,
    crear_modelo_transformer_secuencial
)

VOCAB_SIZE = 5000
MAXLEN = 20
EMBEDDING_DIM = 100

def cargar_corpus(path):
    with open(path, encoding="utf-8") as f:
        texto = f.read().lower()
    return texto

def preparar_datos(texto):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts([texto])
    secuencia_total = tokenizer.texts_to_sequences([texto])[0]

    X = []
    y = []

    for i in range(MAXLEN, len(secuencia_total)):
        X.append(secuencia_total[i-MAXLEN:i])
        y.append(secuencia_total[i])

    X = np.array(X)
    y = np.array(y)
    return X, y, tokenizer

def seleccionar_modelo(nombre, vocab_size, embedding_dim, maxlen):
    nombre = nombre.lower()
    if nombre == "rnn":
        return crear_modelo_rnn_secuencial(vocab_size, embedding_dim, maxlen)
    elif nombre == "cnn":
        return crear_modelo_cnn_secuencial(vocab_size, embedding_dim, maxlen)
    elif nombre == "transformer":
        return crear_modelo_transformer_secuencial(vocab_size, embedding_dim, maxlen)
    else:
        raise ValueError("Modelo no reconocido: usa 'rnn', 'cnn' o 'transformer'")

def entrenar(modelo_nombre, corpus_path, epocas=10):
    print("Cargando y procesando texto...")
    texto = cargar_corpus(corpus_path)
    X, y, tokenizer = preparar_datos(texto)

    print(f"Entrenando modelo {modelo_nombre.upper()}...")
    modelo = seleccionar_modelo(modelo_nombre, VOCAB_SIZE, EMBEDDING_DIM, MAXLEN)
    
    checkpoint = ModelCheckpoint(
        f"modelos/modelo_{modelo_nombre}.keras",
        save_best_only=True,
        monitor='loss',
        mode='min'
    )

    historial = modelo.fit(
        X, y,
        epochs=epocas,
        batch_size=128,
        callbacks=[checkpoint]
    )

    print("Guardando tokenizer y métricas...")
    with open("modelos/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("modelos/historial.pkl", "wb") as f:
        pickle.dump(historial.history, f)

    print("Entrenamiento completo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, default="rnn", help="rnn, cnn, transformer")
    parser.add_argument("--corpus", type=str, default="datos/corpus.txt")
    parser.add_argument("--epocas", type=int, default=10)
    args = parser.parse_args()

    os.makedirs("modelos", exist_ok=True)
    entrenar(args.modelo, args.corpus, args.epocas)
