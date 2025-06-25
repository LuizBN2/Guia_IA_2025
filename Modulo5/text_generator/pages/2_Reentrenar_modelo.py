import streamlit as st
import subprocess

st.header("🔄 Reentrenar Modelo")

modelo_tipo = st.selectbox("Modelo a reentrenar", ["RNN", "CNN", "Transformer"])
epocas = st.slider("Épocas de entrenamiento", 1, 50, 5)

if st.button("Reentrenar"):
    with st.spinner("Entrenando modelo..."):
        proceso = subprocess.Popen(["python", "entrenar_modelos.py", modelo_tipo, str(epocas)])
        proceso.wait()
        st.success("Modelo reentrenado correctamente.")
