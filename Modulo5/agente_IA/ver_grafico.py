import streamlit as st
import base64

# Título
st.title("Visualizador de gráfico generado por el agente")

# Cadena base64 (PEGA AQUÍ la salida del agente, quitando "data:image/png;base64,")
base64_string = """
<PEGA_AQUÍ_TU_CADENA_BASE64>
""".strip()

if base64_string:
    try:
        # Decodificar y mostrar imagen
        img_bytes = base64.b64decode(base64_string)
        st.image(img_bytes, caption="Gráfico generado por el agente", use_column_width=True)
    except Exception as e:
        st.error(f"Error al mostrar la imagen: {e}")
else:
    st.warning("No se proporcionó una cadena base64.")
