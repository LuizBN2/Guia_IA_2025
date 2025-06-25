import streamlit as st
from tensorflow.keras.models import load_model
from tokenizer_utils import cargar_tokenizador, tokenizar_texto
import numpy as np

st.set_page_config(page_title="Clasificar texto", layout="wide")
st.title("🧠 Clasificar Texto")

# Cargar el modelo y el tokenizador
modelo_tipo = st.selectbox("Selecciona el modelo", ["rnn", "cnn", "transformer"])
modelo = load_model(f"modelos/modelo_{modelo_tipo}.keras")
tokenizador = cargar_tokenizador()

# Ingresar texto a clasificar
texto = st.text_area("Ingresa el texto que quieres clasificar:")

if st.button("Clasificar"):
    if texto:
        secuencia_pad = tokenizar_texto(texto, tokenizador)
        prediccion = modelo.predict(secuencia_pad)
        clase_predicha = np.argmax(prediccion, axis=1)
        st.write(f"Texto clasificado como clase: {clase_predicha[0]}")
    else:
        st.error("Por favor, ingresa un texto para clasificar.")
