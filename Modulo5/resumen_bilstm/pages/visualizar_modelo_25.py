from contextlib import redirect_stdout
import pydot
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import io
import os
import time
import numpy as np
from pages.entrenamiento_modelo_25 import entrenamiento
from utils.utils import mostrar_firma_sidebar

st.set_page_config(page_title="Resumen de Noticias", layout="wide")

st.title("Visualización del Modelo BiLSTM para Resumen de Textos")

def sidebar():
    with st.sidebar:
        st.header("🧱 Arquitectura del modelo")
        st.markdown('---')        
        st.page_link('app_resumen_25.py', label='**🏠 Página Principal**')
        st.page_link('pages/entrenamiento_modelo_25.py', label='**🚀 Reentrenamiento**')
        st.page_link('pages/resumir_texto_manual_25.py', label='**📝 Tu resumen**')

sidebar()

def graficas(metricas):
    st.header("Gráfica de Pérdida")
    fig1, ax1 = plt.subplots()
    ax1.plot(metricas['loss'], label='Entrenamiento')
    ax1.plot(metricas['val_loss'], label='Validación')
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Pérdida")
    ax1.set_title("Historial de Pérdida")
    ax1.legend()
    st.pyplot(fig1)

    st.header("Gráfica de Precisión")
    fig2, ax2 = plt.subplots()
    ax2.plot(metricas['accuracy'], label='Entrenamiento')
    ax2.plot(metricas['val_accuracy'], label='Validación')
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Precisión")
    ax2.set_title("Historial de Precisión")
    ax2.legend()
    st.pyplot(fig2)



# === Mostrar Resumen del Modelo ===
def mostrar_resumen_modelo(modelo):
    st.subheader("📋 Resumen del Modelo")
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        modelo.summary()
    st.code(summary_buffer.getvalue(), language='text')


# === Generar archivo DOT personalizado ===
def generar_dot_con_tablas(model, output_path):
    layer_colors = {
        "Embedding": "#dcedc8",
        "Conv1D": "#ffccbc",
        "GlobalMaxPooling1D": "#ffe082",
        "Dense": "#bbdefb",
        "InputLayer": "#f0f0f0"
    }

    def get_shape_safe(tensor):
        try:
            return str(tensor.shape)
        except:
            return "?"

    with open(output_path, "w") as f:
        f.write("digraph G {\n")
        f.write("    rankdir=TB;\n")
        f.write("    concentrate=true;\n")
        f.write("    dpi=200;\n")
        f.write("    splines=ortho;\n")
        f.write("    node [shape=plaintext fontname=Helvetica];\n\n")

        for i, layer in enumerate(model.layers):
            name = layer.name
            tipo = layer.__class__.__name__
            node_id = f"layer_{i}"

            try:
                input_shape = str(layer.input_shape)
            except:
                input_shape = get_shape_safe(layer.input) if hasattr(layer, "input") else "?"

            try:
                output_shape = str(layer.output_shape)
            except:
                output_shape = get_shape_safe(layer.output) if hasattr(layer, "output") else "?"

            color = layer_colors.get(tipo, "#eeeeee")

            label = f"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" BGCOLOR="{color}">
  <TR><TD COLSPAN="2"><B>{name}</B> ({tipo})</TD></TR>
  <TR><TD><FONT POINT-SIZE="10">Input</FONT></TD><TD><FONT POINT-SIZE="10">{input_shape}</FONT></TD></TR>
  <TR><TD><FONT POINT-SIZE="10">Output</FONT></TD><TD><FONT POINT-SIZE="10">{output_shape}</FONT></TD></TR>
</TABLE>>"""
            f.write(f'    {node_id} [label={label}];\n')

        for i in range(1, len(model.layers)):
            f.write(f'    layer_{i-1} -> layer_{i};\n')

        f.write("}\n")


# === Mostrar Visualización de la Arquitectura ===
def mostrar_arquitectura(modelo, dot_output_path, png_output_path):
    st.subheader("🕸️ Arquitectura Visual")

    if not os.path.exists(png_output_path):
        try:
            generar_dot_con_tablas(modelo, dot_output_path)
            (graph,) = pydot.graph_from_dot_file(dot_output_path)
            graph.write_png(png_output_path)
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar el diagrama: {e}")

    if os.path.exists(png_output_path):
        st.image(png_output_path, caption="Estructura de la red neuronal", use_column_width=True)

        with open(png_output_path, "rb") as file:
            st.download_button(
                label="💾 Guardar imagen del diagrama",
                data=file,
                file_name="modelo_arquitectura.png",
                mime="image/png"
            )

def cargar_modelo(ruta):
    # Cargar el modelo
    try:
        model = load_model("models/modelo_resumen_bilstm_25.keras")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()


# === Interfaz Principal ===
def main():
    # Cargar el historial
    if os.path.exists("data/metricas_entrenamiento_25.csv"):
        metricas = pd.read_csv("data/metricas_entrenamiento_25.csv")
        graficas(metricas)
    else:
        st.error("No se encontró el archivo 'data/metricas_entrenamiento_25.csv'")
        st.stop()
    modelo = cargar_modelo("models/modelo_resumen_bilstm_25.keras")
    mostrar_resumen_modelo(modelo)
    mostrar_arquitectura(modelo, "images/modelo_coloreado.dot", "images/modelo_coloreado.png")

# === Lanzar App ===
if __name__ == '__main__':
    main()
    mostrar_firma_sidebar()