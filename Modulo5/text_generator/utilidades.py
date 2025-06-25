import pickle
import streamlit as st

def guardar_pickle(objeto, ruta):
    with open(ruta, 'wb') as f:
        pickle.dump(objeto, f)

def cargar_pickle(ruta):
    with open(ruta, 'rb') as f:
        return pickle.load(f)

def barra_progreso(texto, pasos):
    with st.spinner(texto):
        for _ in pasos:
            st.progress(_ / len(pasos))
