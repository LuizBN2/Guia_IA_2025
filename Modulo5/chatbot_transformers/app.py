import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Chatbot Transformers", layout="centered")

# Cargar modelo
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# Inicializar estado de sesión
if "history_ids" not in st.session_state:
    st.session_state.history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []
if "responses" not in st.session_state:
    st.session_state.responses = []

# Título
st.title("🤖 Chatbot con Transformers")
st.caption("Basado en DialoGPT de Microsoft")

# Botón para reiniciar conversación
if st.button("🔄 Reiniciar conversación"):
    st.session_state.history_ids = None
    st.session_state.past_inputs = []
    st.session_state.responses = []
    st.experimental_rerun()

# Entrada del usuario
user_input = st.chat_input("Escribe tu mensaje...")

if user_input:
    with st.spinner("Escribiendo respuesta..."):
        # Codificar entrada
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        # Combinar con historial
        bot_input_ids = torch.cat([st.session_state.history_ids, new_input_ids], dim=-1) if st.session_state.history_ids is not None else new_input_ids

        # Generar respuesta
        st.session_state.history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        output = st.session_state.history_ids[:, bot_input_ids.shape[-1]:][0]
        response = tokenizer.decode(output, skip_special_tokens=True)

        # Guardar historial
        st.session_state.past_inputs.append(user_input)
        st.session_state.responses.append(response)

# Mostrar historial de conversación
for user_msg, bot_msg in zip(st.session_state.past_inputs, st.session_state.responses):
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
