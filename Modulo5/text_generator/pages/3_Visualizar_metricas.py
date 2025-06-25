import streamlit as st
from utilidades import cargar_pickle
import matplotlib.pyplot as plt

st.header("📈 Visualización de Métricas")

modelo_tipo = st.selectbox("Modelo", ["RNN", "CNN", "Transformer"])
historial = cargar_pickle(f"modelos/historial_{modelo_tipo.lower()}.pkl")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(historial['loss'], label='Pérdida')
ax[0].plot(historial['val_loss'], label='Val Pérdida')
ax[0].legend()
ax[0].set_title("Pérdida")

ax[1].plot(historial['accuracy'], label='Precisión')
ax[1].plot(historial['val_accuracy'], label='Val Precisión')
ax[1].legend()
ax[1].set_title("Precisión")

st.pyplot(fig)

st.subheader("Resumen del modelo")
with open(f"modelos/resumen_{modelo_tipo.lower()}.txt") as f:
    resumen = f.read()
st.code(resumen, language='bash')
