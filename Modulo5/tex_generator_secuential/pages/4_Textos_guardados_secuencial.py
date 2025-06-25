import streamlit as st
import os

st.set_page_config(page_title="Textos Guardados", layout="wide")
st.title("📂 Textos Generados y Guardados")

directorio = "textos_generados"
os.makedirs(directorio, exist_ok=True)

archivos = [f for f in os.listdir(directorio) if f.endswith(".txt")]

if not archivos:
    st.info("No hay textos guardados aún.")
else:
    for archivo in archivos:
        ruta = os.path.join(directorio, archivo)
        with open(ruta, encoding="utf-8") as f:
            contenido = f.read()

        with st.expander(f"📄 {archivo}"):
            st.text_area("Contenido:", contenido, height=200)
            st.download_button(
                label="⬇️ Descargar",
                data=contenido,
                file_name=archivo,
                mime="text/plain"
            )
