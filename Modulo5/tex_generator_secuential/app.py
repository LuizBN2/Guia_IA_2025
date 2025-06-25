import streamlit as st

st.set_page_config(page_title="Generador de Texto", layout="wide")
st.title("🧠 Generador de Texto en Español con Deep Learning")

st.markdown("""
Bienvenido al generador de texto entrenado con redes neuronales.  
Navega por las pestañas de la izquierda para:

1. **Generar texto** a partir de un modelo y un umbral.
2. **Reentrenar modelos** con nuevas épocas.
3. **Visualizar métricas y arquitecturas** de modelos.
4. **Guardar y revisar textos generados**.

---
""")
