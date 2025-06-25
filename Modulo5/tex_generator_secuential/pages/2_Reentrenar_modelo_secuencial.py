import streamlit as st
import subprocess
import threading
import os
from pathlib import Path

st.set_page_config(page_title="Reentrenar Modelo", layout="wide")
st.title("🔁 Reentrenamiento del Modelo Generador")

st.markdown("Selecciona el modelo y ajusta las épocas para reentrenar usando el corpus disponible.")

modelos = {
    "RNN": "rnn",
    "CNN": "cnn",
    "Transformer": "transformer"
}

modelo_seleccionado = st.selectbox("Tipo de modelo", list(modelos.keys()))
epocas = st.slider("Número de épocas", 1, 50, 10)
corpus_path = "datos/corpus.txt"

progreso = st.empty()
estado = st.empty()

def entrenar_en_subproceso(modelo, epocas):
    comando = [
        "python",
        "entrenar_modelos_secuencial.py",
        "--modelo", modelo,
        "--corpus", corpus_path,
        "--epocas", str(epocas)
    ]

    proceso = subprocess.Popen(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    while True:
        output = proceso.stdout.readline()
        if output == '' and proceso.poll() is not None:
            break
        if output:
            estado.text(output.strip())

    progreso.empty()
    estado.success("✅ Entrenamiento finalizado")

if st.button("Reentrenar modelo"):
    if not Path(corpus_path).exists():
        st.error(f"El archivo del corpus no existe en `{corpus_path}`.")
    else:
        progreso.progress(0)
        progreso.progress(20)
        estado.info("🔄 Entrenando modelo... esto puede tardar unos minutos.")

        hilo = threading.Thread(
            target=entrenar_en_subproceso,
            args=(modelos[modelo_seleccionado], epocas)
        )
        hilo.start()
