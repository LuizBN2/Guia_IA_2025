import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import io
import os
from PIL import Image

st.set_page_config(page_title="Visualizar Métricas", layout="wide")
st.title("📊 Visualización de Métricas y Arquitectura")

modelos_disponibles = {
    "RNN": "modelo_rnn.keras",
    "CNN": "modelo_cnn.keras",
    "Transformer": "modelo_transformer.keras"
}

modelo_seleccionado = st.selectbox("Selecciona el modelo", list(modelos_disponibles.keys()))
modelo_path = f"modelos/{modelos_disponibles[modelo_seleccionado]}"

try:
    modelo = load_model(modelo_path)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Mostrar resumen del modelo
st.subheader("📄 Resumen del Modelo")
summary_io = io.StringIO()
modelo.summary(print_fn=lambda x: summary_io.write(x + "\n"))
st.code(summary_io.getvalue(), language="text")

# Mostrar arquitectura con colores
st.subheader("🧠 Arquitectura del Modelo (Diagrama)")
imagen_path = f"modelos/arquitectura_{modelo_seleccionado.lower()}.png"
try:
    plot_model(modelo, to_file=imagen_path, show_shapes=True, show_layer_names=True, dpi=100)
    st.image(Image.open(imagen_path), caption="Diagrama del modelo", use_column_width=True)
except Exception as e:
    st.warning(f"No se pudo generar el diagrama: {e}")

# Mostrar métricas
st.subheader("📈 Métricas de Entrenamiento")

try:
    with open("modelos/historial.pkl", "rb") as f:
        historial = pickle.load(f)
    df = pd.DataFrame(historial)
except Exception as e:
    st.error(f"No se pudo cargar el historial de métricas: {e}")
    st.stop()

# Gráficos
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Pérdida (Loss)**")
    fig, ax = plt.subplots()
    df["loss"].plot(ax=ax, color="tomato", linewidth=2)
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

with col2:
    if "accuracy" in df.columns:
        st.markdown("**Precisión (Accuracy)**")
        fig2, ax2 = plt.subplots()
        df["accuracy"].plot(ax=ax2, color="seagreen", linewidth=2)
        ax2.set_xlabel("Épocas")
        ax2.set_ylabel("Accuracy")
        st.pyplot(fig2)
    else:
        st.info("Este modelo no tiene métrica de precisión.")

