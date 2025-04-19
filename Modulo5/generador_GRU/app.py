# app.py
import streamlit as st
from model_utils import generar_texto, cargar_metricas, entrenar_modelo
from datetime import datetime
import subprocess
import os
import time
import pandas as pd
import json

st.set_page_config(page_title="Generador de Texto", layout="centered")

st.title("🧠 Generador de Texto con Modelos RNN / Transformer")

def verificar_modelo_actualizado(modelo):
    modelo_path = f"modelos/modelo_{modelo}.keras"
    if os.path.exists(modelo_path):
        last_modified = os.path.getmtime(modelo_path)
        return time.ctime(last_modified)
    return None

tab1, tab2, tab3, tab4 = st.tabs(["Generador", "Entrenamiento", "Métricas", "Descargas"])

# ----- TAB 1: Generador de texto -----
with tab1:
    st.subheader("Generar texto")
    modelo_seleccionado = st.selectbox("Selecciona el modelo", ["GRU", "LSTM", "Transformer"])
    seed_text = st.text_input("Texto inicial", "once upon a time")
    next_words = st.slider("Número de palabras", 10, 100, 30)
    temperature = st.slider("Temperatura", 0.2, 2.0, 1.0, step=0.1)

    if st.button("Generar texto"):
        with st.spinner("Generando texto..."):
            texto = generar_texto(
                seed_text, next_words, temperature,
                modelo_seleccionado.lower(),
                st.progress(0), st.empty()
            )

        st.success("✅ Texto generado exitosamente.")
        st.markdown("### Resultado:")
        st.text(texto)
        st.download_button(
            label="💾 Descargar texto",
            data=texto,
            file_name=f"texto_generado_{modelo_seleccionado.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ----- TAB 2: Entrenamiento -----
# ----- TAB 2: Entrenamiento -----
with tab2:
    st.title("📊 Reentrenamiento y Visualización de Modelos")

    modelo_nuevo = st.selectbox("Modelo a reentrenar", ["GRU", "LSTM", "Transformer"])
    epocas = st.slider("Número de épocas", min_value=1, max_value=50, value=10)
    nombre_csv = f"historiales/historial_{modelo_nuevo.lower()}.csv"

    progreso_path = "historiales/progreso_entrenamiento.json"

    # Iniciar reentrenamiento
    if st.button("🚀 Iniciar reentrenamiento"):
        st.info(f"🛠️ Entrenando modelo {modelo_nuevo.upper()} en segundo plano...")

        # Eliminar el JSON anterior si existe
        if os.path.exists(progreso_path):
            os.remove(progreso_path)

        comando = ["python", "entrenar_todos.py", modelo_nuevo.lower(), str(epocas)]
        
        
        
        
        try:
            try:
                result = subprocess.run(comando, capture_output=True, text=True, check=True)
                st.success("✅ Entrenamiento iniciado.")
                st.text(result.stdout)  # Mostrar la salida estándar del proceso
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Error al ejecutar el script: {e.stderr}")
            # Después de lanzar el comando
            time.sleep(3)

            if os.path.exists("historiales/log.txt"):
                with open("historiales/log.txt", "r") as f:
                    st.code(f.read(), language="text")
            else:
                st.warning("⚠️ El script no se ejecutó. Revisa la ruta o permisos.")




            with open(f"historiales/last_train_{modelo_nuevo.lower()}.txt", "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Esperar a que se cree el archivo de progreso
            with st.spinner("⏳ Esperando inicio del entrenamiento..."):
                for _ in range(30):  # 30 segundos máx
                    if os.path.exists(progreso_path):
                        break
                    time.sleep(1)
                else:
                    st.error("⚠️ No se detectó inicio del entrenamiento tras 30 segundos.")
                    st.stop()

            st.success("✅ Entrenamiento iniciado.")
        except Exception as e:
            st.error(f"❌ Error al lanzar el entrenamiento: {e}")
            st.stop()

    # Mostrar progreso de entrenamiento en tiempo real
    st.markdown("---")
    st.subheader("📡 Progreso en tiempo real")

    progreso_container = st.empty()
    barra_container = st.empty()

    if os.path.exists(progreso_path):
        while True:
            try:
                with open(progreso_path, "r") as f:
                    progreso = json.load(f)

                epoch = progreso.get("epoch", 0)
                total = progreso.get("total_epochs", epocas)
                acc = progreso.get("accuracy", 0)
                val_acc = progreso.get("val_accuracy", 0)
                loss = progreso.get("loss", 0)
                val_loss = progreso.get("val_loss", 0)

                progreso_container.markdown(
                    f"📈 **Época {epoch}/{total}**  \n"
                    f"✅ Accuracy: `{acc:.4f}` | 🔍 Val_Accuracy: `{val_acc:.4f}`  \n"
                    f"❌ Loss: `{loss:.4f}` | 🔎 Val_Loss: `{val_loss:.4f}`"
                )
                barra_container.progress(epoch / total)

                if epoch >= total:
                    st.success("🏁 Entrenamiento finalizado.")
                    break
                time.sleep(2)
            except Exception as e:
                st.warning(f"Esperando datos de progreso... ({e})")
                time.sleep(2)
    else:
        st.info("🔁 Inicia un entrenamiento para ver el progreso.")



# ----- TAB 3: Visualización de métricas -----
with tab3:
    st.subheader("Historial de entrenamiento")
    modelo = st.selectbox("Modelo", ["GRU", "LSTM", "Transformer"])
    metricas = cargar_metricas(modelo.lower())
    if metricas is not None:
        st.line_chart(metricas[["loss", "accuracy"]])
    else:
        st.warning("Aún no hay métricas disponibles para este modelo.")

# ----- TAB 4: Descargas -----
with tab4:
    st.subheader("Archivos disponibles")
    st.markdown("📁 Puedes acceder a los modelos, tokenizer y corpus manualmente en la carpeta del proyecto.")
