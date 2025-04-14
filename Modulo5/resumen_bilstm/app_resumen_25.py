import streamlit as st
import pandas as pd
from pages.resumen_modelo_25 import resumir_noticia
from utils.utils import mostrar_firma_sidebar
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import os

st.set_page_config(page_title="Resumen de Noticias", layout="wide")

st.title("🧠 Generador de Resúmenes con BiLSTM")

# Cargar modelo y métricas
modelo = load_model("models/modelo_resumen_bilstm_25.keras")

# Cargar tokenizer
with open("models/tokenizer_resumen_25.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Cargar dataset
df = pd.read_csv("data/df_total.csv", encoding='utf-8')
df['titulo'] = df['news'].apply(lambda x: x.split('.')[0][:100])  # asumimos que no hay columna título


with st.sidebar:
    st.header("🔍 Selecciona una noticia")
    titulo_seleccionado = st.selectbox("Títulos disponibles:", df['titulo'])
    umbral = st.slider("Ajustar umbral de relevancia", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

def sidebar():
    with st.sidebar:        
        st.header("🔍 Entrenamiento del modelo")
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
        st.markdown('---')
        st.page_link('pages/entrenamiento_modelo_25.py', label='**🚀 Reentrenamiento**')
        st.page_link('pages/visualizar_modelo_25.py', label='**🧱 Arquitectura**')
        st.page_link('pages/resumir_texto_manual_25.py', label='**📝 Tu resumen**')

mostrar_firma_sidebar()

# ----------------- Selección de noticia -----------------


# Obtener la noticia seleccionada
noticia = df[df['titulo'] == titulo_seleccionado]['news'].values[0]

st.subheader("📰 Noticia Original")
st.write(noticia)

sidebar()

# ----------------- Generar resumen -----------------
if st.button("📌 Generar Resumen"):
    resumen, word = resumir_noticia(noticia, modelo, tokenizer, umbral)
    st.subheader("📝 Resumen Generado")
    if resumen:
        for i, oracion in enumerate(resumen, 1):
            st.markdown(f"{oracion}\n\n")
    else:
        st.warning("⚠️ No se encontraron oraciones relevantes para este umbral.")

    # ----------------- WordCloud del resumen -----------------
    if word:
        st.subheader("☁️ WordCloud del Resumen")
        resumen_texto = " ".join(word)
        wc = WordCloud(background_color='white', width=800, height=400).generate(resumen_texto)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)


