import streamlit as st
from utilidades import cargar_pickle, guardar_pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
from generar_texto import cargar_modelo_y_tokenizer, generar_texto_desde_corpus, g

st.header("✍️ Generar Texto")

modelo_tipo = st.selectbox("Seleccionar modelo", ["RNN", "CNN", "Transformer"])
umbral = st.slider("Umbral de predicción", 0.0, 1.0, 0.5, 0.05)



modelo, tokenizer = cargar_modelo_y_tokenizer(modelo_tipo)
texto_final, oraciones_generadas, predicciones = generar_texto_desde_corpus(
    corpus_path="datos/corpus.txt",
    modelo=modelo,
    tokenizer=tokenizer,
    umbral=umbral
)
st.text_area("Texto generado", texto_final, height=250)


if st.button("💾 Guardar texto generado"):
    nombre_archivo = guardar_texto_generado(texto_generado, modelo_seleccionado)
    st.success(f"Texto guardado como: {nombre_archivo}")
