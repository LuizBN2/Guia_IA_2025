import streamlit as st
import json
import os
import random

# Archivo para guardar conocimientos
ARCHIVO_CONOCIMIENTO = "conocimiento.json"

# Cargar conocimiento o crear uno nuevo
if os.path.exists(ARCHIVO_CONOCIMIENTO):
    with open(ARCHIVO_CONOCIMIENTO, "r", encoding="utf-8") as f:
        conocimiento = json.load(f)
else:
    conocimiento = {
        "respuestas_bot": {
            "hola": "Hola, ¿cómo estás?",
            "¿cómo estas?": "Muy bien y tú, preparado para resolver tus inquietudes sobre el bootcamp.",
            "¿cuántos años tienes?": "Tengo unas pocas horas de ser creado, tus profes me diseñaron para conocerte mejor.",
            "¿a qué te dedicas?": "Soy un bot de AI, diseñado para ofrecerte respuestas sencillas sobre el curso.",
            "¿qué estudias?": "Inteligencia Artificial, nivel Explorador, ¿y tú?",
            "¿qué aprenderé en este curso?": "Todo lo relacionado sobre el Procesamiento de Lenguaje Natural a través de Inteligencia Artificial.",
            "¿cómo se llama el curso?": "Inteligencia Artificial Explorador",
            "¿qué días tenemos clase?": "Los sábados y domingos hasta el mes de junio.",
            "¿cuál es el horario de clases?": "De 7 de la mañana, hasta las 5 de la tarde.",
            "¿dónde se dictará el bootcamp o curso?": "En la Universidad de Medellín."
        },
        "preguntas_al_usuario": [
            "¿Qué expectativas tienes del curso?",
            "¿Dónde trabajas?",
            "¿Sabes programar?",
            "¿En qué lenguajes has programado?",
            "¿Cómo conociste el curso?",
            "¿Qué opinas del horario del bootcamp?",
            "¿Para qué te servirá este curso o bootcamp?",
            "¿Has participado antes de un bootcamp?",
            "¿Cuál es tu color favorito?",
            "¿Cuál es tu música favorita?",
            "¿Qué opinas sobre la inteligencia artificial?"
        ]
    }

# Guardar conocimiento
def guardar_conocimiento():
    with open(ARCHIVO_CONOCIMIENTO, "w", encoding="utf-8") as f:
        json.dump(conocimiento, f, indent=4, ensure_ascii=False)

st.set_page_config(page_title="Chatbot IA que Aprende", page_icon="🧠")
st.title("🧠 Chatbot Inteligente - Aprende de ti")
st.markdown("¡Hola! Estoy listo para responderte y aprender de tus nuevas preguntas.")

# Inicializar historial de conversación
if "historial" not in st.session_state:
    st.session_state.historial = []

user_input = st.text_input("Escribe tu mensaje aquí:")

if user_input:
    user_input_clean = user_input.strip().lower()
    st.session_state.historial.append(("Tú", user_input))

    if user_input_clean in conocimiento["respuestas_bot"]:
        respuesta = conocimiento["respuestas_bot"][user_input_clean]
    else:
        respuesta = "No sé la respuesta... ¿Quieres enseñármela?"
        if st.session_state.get("esperando_respuesta") != user_input_clean:
            st.session_state.esperando_respuesta = user_input_clean
        else:
            nueva_respuesta = st.text_input(f"¿Cuál sería una buena respuesta para '{user_input}'?", key="nueva_respuesta")
            if nueva_respuesta:
                conocimiento["respuestas_bot"][user_input_clean] = nueva_respuesta
                guardar_conocimiento()
                respuesta = "¡Gracias! He aprendido algo nuevo. 😊"
                st.session_state.esperando_respuesta = None

    st.session_state.historial.append(("Bot", respuesta))

# Mostrar historial
st.subheader("🗨️ Conversación")
for rol, mensaje in st.session_state.historial:
    st.markdown(f"**{rol}:** {mensaje}")

# Pregunta aleatoria del bot al usuario
if st.button("🤖 Hazme una pregunta"):
    pregunta = random.choice(conocimiento["preguntas_al_usuario"])
    st.session_state.historial.append(("Bot", pregunta))

import subprocess

def hacer_backup_en_github():
    try:
        subprocess.run(["git", "add", "conocimiento.json"], check=True)
        subprocess.run(["git", "commit", "-m", "🧠 Backup automático del conocimiento del chatbot"], check=True)
        subprocess.run(["git", "push"], check=True)
        st.success("Backup realizado en GitHub.")
    except Exception as e:
        st.warning(f"No se pudo hacer el backup automático: {e}")

# Botón para guardar y subir a GitHub
if st.button("📤 Hacer backup en GitHub"):
    guardar_conocimiento()
    hacer_backup_en_github()