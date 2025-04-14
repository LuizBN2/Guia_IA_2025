import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from pages.resumen_modelo_25 import resumir_noticia
from utils.utils import mostrar_firma_sidebar
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Resumen de Noticias", layout="wide")

st.title("📝 Ingresar texto para resumir")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Parámetros")
    umbral = st.slider("Ajustar umbral de relevancia", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    st.page_link('app_resumen_25.py', label='**🏠 Página Principal**')
    st.page_link('pages/entrenamiento_modelo_25.py', label='**🚀 Reentrenamiento**')
    st.page_link('pages/visualizar_modelo_25.py', label='**🧱 Arquitectura**')
    mostrar_firma_sidebar()

# ------------------ Cargar modelo y tokenizer ------------------
modelo = load_model("models/modelo_resumen_bilstm_25.keras")

with open("models/tokenizer_resumen_25.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ------------------ Entrada de texto ------------------
st.subheader("✏️ Escribe o pega un texto")
texto_usuario = st.text_area("Introduce aquí la noticia o el texto completo que deseas resumir:", height=300)

if st.button("📌 Generar Resumen"):
    if texto_usuario.strip():
        resumen, palabras_clave = resumir_noticia(texto_usuario, modelo, tokenizer, umbral)
        st.subheader("📝 Resumen Generado")
        if resumen:
            for i, oracion in enumerate(resumen, 1):
                st.markdown(f"{oracion}")
        else:
            st.warning("⚠️ No se encontraron oraciones relevantes para este umbral.")
        
         # ------------------ WordCloud ------------------
        if palabras_clave:
            st.subheader("☁️ WordCloud del Resumen")
            texto_resumen = " ".join(palabras_clave)
            wc = WordCloud(background_color='white', width=800, height=400).generate(texto_resumen)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    else:
        st.error("❌ Por favor ingresa un texto antes de generar el resumen.")
