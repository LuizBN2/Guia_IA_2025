# model_utils.py

import os
with open("historiales/log.txt", "a") as f:
    f.write("🟢 Script fue llamado desde Streamlit\n")


import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import gc
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import streamlit as st

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
MAX_VOCAB_SIZE = 5000
MODELOS_PATH = "modelos/"
HISTORIALES_PATH = "historiales/"
TOKENIZER_PATH = "tokenizer.pkl"
CORPUS_PATH = "corpus_completo.txt"

# ===============================
# FUNCIONES
# ===============================

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generar_texto(seed_text, next_words, temperature, modelo_nombre, st_progress=None, st_status=None):
    modelo = load_model(f"{MODELOS_PATH}modelo_{modelo_nombre}.keras")
    
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    max_seq_len = modelo.input_shape[1]

    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_seq_len:]  # recorte
        token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
        
        preds = modelo.predict(token_list, verbose=0)[0]
        next_index = sample_with_temperature(preds, temperature)
        output_word = tokenizer.index_word.get(next_index, '')
        seed_text += " " + output_word

        # Actualizar barra de progreso si estamos en Streamlit
        if st_progress is not None:
            st_progress.progress((i + 1) / next_words)
        if st_status is not None:
            st_status.text(f"🧠 Generando palabra {i + 1}/{next_words}: {output_word}")

    return seed_text


def cargar_metricas(modelo_nombre):
    csv_path = f"{HISTORIALES_PATH}historial_{modelo_nombre}.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return None


def entrenar_modelo(modelo_nombre, epocas, st_progress=None, st_status=None):
    # Verifica si se han recibido las variables de progreso
    if st_progress is None or st_status is None:
        raise ValueError("Se requiere un objeto 'st_progress' y 'st_status' para mostrar el progreso.")

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        texto = f.read().lower()

    sequences = []
    total_words = len(tokenizer.word_index) + 1

    tokens = tokenizer.texts_to_sequences([texto])[0]
    for i in range(1, len(tokens)):
        n_gram_sequence = tokens[:i+1]
        sequences.append(n_gram_sequence)

    max_seq_len = max([len(x) for x in sequences])
    sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

    X = sequences[:, :-1]
    y = to_categorical(sequences[:, -1], num_classes=total_words)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_seq_len-1))

    if modelo_nombre == "gru":
        model.add(GRU(150))
    elif modelo_nombre == "lstm":
        model.add(LSTM(150))
    else:
        raise ValueError("Modelo no soportado para entrenamiento desde la app")

    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Mostrar barra de progreso en Streamlit
    st_status.text(f"🧠 Iniciando el reentrenamiento de {modelo_nombre}...")
    st_progress.progress(0)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epocas, verbose=0, 
                        callbacks=[TqdmCallback(verbose=1)])

    # Actualizar progreso
    st_progress.progress(100)
    st_status.text("🏁 Entrenamiento finalizado.")

    model.save(f"{MODELOS_PATH}modelo_{modelo_nombre}.keras")

    df_hist = pd.DataFrame(history.history)
    df_hist.to_csv(f"{HISTORIALES_PATH}historial_{modelo_nombre}.csv", index=False)

    del model, X, y, X_train, y_train, X_val, y_val
    gc.collect()

    return df_hist
