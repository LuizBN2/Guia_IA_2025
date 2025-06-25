import sys
import asyncio

# Solución para el error NotImplementedError en Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nest_asyncio
nest_asyncio.apply()


import streamlit as st
from app_2 import agent  # reutilizamos tu agente ya configurado

st.set_page_config(page_title="Agente IA Conversacional", layout="wide")

st.title("🤖 Agente IA Conversacional")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Escribe tu mensaje aquí...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Pensando..."):
        async def call_agent():
            async with agent.run_mcp_servers():
                try:
                    result = await agent.run(user_input)
                    return result.output
                except Exception as e:
                    return f"Error: {e}"

        response = asyncio.run(call_agent())
        st.session_state.messages.append({"role": "agent", "content": response})

# Mostrar conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["content"].startswith("data:image/png;base64"):
            st.image(msg["content"])
        else:
            st.markdown(msg["content"])
