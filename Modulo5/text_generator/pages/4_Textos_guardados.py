import streamlit as st
import os

st.header("📚 Textos Generados Guardados")

carpeta = "datos/textos_generados"
textos = os.listdir(carpeta)

for archivo in textos:
    st.subheader(archivo)
    with open(os.path.join(carpeta, archivo), encoding='utf-8') as f:
        st.text_area("", f.read(), height=150)
