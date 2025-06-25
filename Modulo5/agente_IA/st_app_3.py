import streamlit as st
import requests

st.set_page_config(page_title="Agente IA Conversacional", layout="wide")
st.title("🤖 Agente IA Conversacional")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Escribe tu mensaje aquí...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Pensando..."):
        try:
            response = requests.post("http://127.0.0.1:8000/chat", json={"content": user_input})
            agent_reply = response.json().get("response", "Sin respuesta del agente.")
        except Exception as e:
            agent_reply = f"Error: {e}"

        st.session_state.messages.append({"role": "agent", "content": agent_reply})

# Mostrar conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["content"].startswith("data:image/png;base64"):
            st.image(msg["content"])
        else:
            st.markdown(msg["content"])
