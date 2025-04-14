import streamlit as st
import pandas as pd
import numpy as np
import nltk
import pickle
import os
import nltk

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import Callback
from nltk.corpus import stopwords
import re

from utils.utils import mostrar_firma_sidebar

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

st.set_page_config(page_title="Resumen de Noticias", layout="wide")

st.title("🔁 Reentrenar modelo BiLSTM para resumen de noticias")

def sidebar():
    with st.sidebar:
        st.header("🚀 Entrenamiento del modelo")
        st.markdown('---')
    
        st.subheader("📊 Métricas del Modelo")

        metrics_path = "data/metricas_entrenamiento_25.csv"
        if os.path.exists(metrics_path):
            metricas_df = pd.read_csv(metrics_path)
            #st.info(metricas_df)
            #st.line_chart(metricas_df[['loss', 'val_loss']])
            st.line_chart(metricas_df[['accuracy', 'val_accuracy']])
        else:
            st.info("No se encontraron métricas guardadas.")

        st.page_link('app_resumen_25.py', label='**🏠 Página Principal**')
        st.page_link('pages/visualizar_modelo_25.py', label='**🧱 Arquitectura**')
        st.page_link('pages/resumir_texto_manual_25.py', label='**📝 Tu resumen**')
        mostrar_firma_sidebar()
        

# Función para limpiar texto y eliminar stopwords
def limpiar_y_filtrar(texto):
    # Solo letras, minúsculas
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]', '', texto)
    palabras = texto.lower().split()
    palabras_filtradas = [p for p in palabras if p not in stop_words]
    return ' '.join(palabras_filtradas)


#función de entrenamiento
def entrenamiento(epochs):
    print('hola')
    # Botón de entrenamiento
    
    st.info("Cargando datos y procesando...")

    df = pd.read_csv("data/df_total.csv", encoding='utf-8')
    docs = df['news'].dropna().tolist()

    # Crear dataset etiquetado
    X, y = [], []
    for doc in docs:
        oraciones = nltk.sent_tokenize(doc, language='spanish')
        if len(oraciones) < 4:
            continue
        for i, oracion in enumerate(oraciones):
            X.append(oracion)
            y.append(1 if i < 3 else 0)

    # Tokenización
    # Aplicar limpieza a todos los textos
    X_filtrado = [limpiar_y_filtrar(oracion) for oracion in X]

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_filtrado)
    sequences = tokenizer.texts_to_sequences(X_filtrado)
    X_pad = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = np.array(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    # Modelo
    modelo = Sequential()
    modelo.add(Embedding(input_dim=num_words, output_dim=128, input_length=max_len))
    modelo.add(Bidirectional(LSTM(64, return_sequences=False)))
    modelo.add(Dense(1, activation='sigmoid'))

    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    st.info("Entrenando modelo...")
    # Barra de progreso y estado
    progress_bar = st.progress(0)
    status_text = st.empty()

    class StreamlitProgressCallback(Callback):
        def __init__(self, epochs):
            super().__init__()
            self.epochs = epochs

        def on_epoch_end(self, epoch, logs=None):
            progress = int((epoch + 1) / self.epochs * 100)
            progress_bar.progress(progress)
            status_text.text(f"Época {epoch + 1}/{self.epochs} completada")
    history = modelo.fit(
                        X_train, 
                        y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_test, y_test), 
                        verbose=0,
                        callbacks=[StreamlitProgressCallback(epochs)]
                        )
    
    st.success("Entrenamiento finalizado ✅")

    # Mostrar resultados
    df_hist = pd.DataFrame(history.history)
    st.subheader("📈 Métricas del entrenamiento")
    st.line_chart(df_hist[['loss', 'val_loss']])
    st.line_chart(df_hist[['accuracy', 'val_accuracy']])

    # Guardar métricas temporalmente
    df_hist.to_csv("data/metricas_entrenamiento_25.csv", index=False)
    modelo_g = save_model(modelo, "models/modelo_resumen_bilstm_25.keras")
    with open("models/tokenizer_resumen_25.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("models/historial_entrenamiento_25.pkl", "wb") as g:
        pickle.dump(history.history, g)
    
    if st.button('✅ Aceptar'):
        st.rerun()
    return modelo


# Parámetros
num_words = 10000
max_len = 40
batch_size = 32
sidebar()

# Epochs por slider
epochs = st.slider("Selecciona la cantidad de épocas", min_value=1, max_value=20, value=5)


if st.button("🚀 Entrenar modelo"):
    entrenamiento(epochs)



    
