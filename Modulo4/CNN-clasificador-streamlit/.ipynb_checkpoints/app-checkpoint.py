import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Titulo de la app
st.title("🧠 Clasificador de Texto con Deep Learning")

# Cargar el modelo y tokenizer
modelo = load_model('modelo21.keras', compile=False)
modelo.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Cargar datos y tokenizer
df = pd.read_csv('data/df_total.csv')
df['target'] = df['Type'].astype('category').cat.codes
idx2label = dict(enumerate(df['Type'].astype('category').cat.categories))

# Tokenizer
max_vocab_size = 30000
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(df['news'])

# Secuencia de padding
sequences = tokenizer.texts_to_sequences(df['news'])
data = pad_sequences(sequences)
T = data.shape[1]

# Entrada del usuario
st.markdown("## 🔍 Clasificación de texto")
option = st.radio("Selecciona el método de entrada:", ('Escribir texto', 'Subir archivo .txt'))

if option == 'Escribir texto':
    user_input = st.text_area("Ingresa el texto a clasificar:")
elif option == 'Subir archivo .txt':
    uploaded_file = st.file_uploader("Sube tu archivo .txt", type="txt")
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode('utf-8')
    else:
        user_input = ""

if user_input:
    # Procesar entrada
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=T)
    pred = modelo.predict(padded)
    probs = tf.nn.softmax(pred[0]).numpy()
    pred_idx = np.argmax(probs)
    pred_label = idx2label[pred_idx]

    st.success(f"🔎 Predicción: **{pred_label}**")

    # Mostrar probabilidades
    st.subheader("📊 Probabilidades por clase")
    df_probs = pd.DataFrame({
        'Clase': list(idx2label.values()),
        'Probabilidad': probs
    }).sort_values('Probabilidad', ascending=False)
    st.dataframe(df_probs, use_container_width=True)

    fig, ax = plt.subplots()
    ax.barh(df_probs['Clase'], df_probs['Probabilidad'], color='skyblue')
    ax.invert_yaxis()
    st.pyplot(fig)

# Gráficos de entrenamiento
st.markdown("---")
st.subheader("📉 Evolución del entrenamiento")

try:
    with open('historial_entrenamiento.pkl', 'rb') as f:
        history = pickle.load(f)

    fig1, ax1 = plt.subplots()
    ax1.plot(history['loss'], label='Pérdida del Entrenamiento', color='blue')
    ax1.plot(history['val_loss'], label='Pérdida del set de prueba', color='orange')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Evolución del entrenamiento')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    if 'accuracy' in history and 'val_accuracy' in history:
        fig2, ax2 = plt.subplots()
        ax2.plot(history['accuracy'], label='Precisión del Entrenamiento', color='red')
        ax2.plot(history['val_accuracy'], label='Precisión del set de prueba', color='black')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Precisión')
        ax2.set_title('Evolución del entrenamiento')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

except FileNotFoundError:
    st.info("ℹ️ No se encontró el archivo `historial_entrenamiento.pkl`. Entrena el modelo y guarda el historial para visualizar estas gráficas.")

# Matriz de confusión
st.markdown("---")
st.subheader("🧪 Matriz de Confusión")

try:
    _, df_test = train_test_split(df, test_size=0.3, random_state=42)
    sequences_test = tokenizer.texts_to_sequences(df_test['news'])
    data_test = pad_sequences(sequences_test, maxlen=T)
    Y_test = df_test['target'].values

    y_pred_logits = modelo.predict(data_test)
    y_pred = np.argmax(y_pred_logits, axis=1)

    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(idx2label.values()))

    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax_cm, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax_cm.set_title("Matriz de Confusión sobre el set de prueba")
    st.pyplot(fig_cm)

except Exception as e:
    st.warning(f"⚠️ Error al generar la matriz de confusión: {e}")

# Score final del modelo
st.markdown("### 📌 Rendimiento general del modelo")

try:
    loss, acc = modelo.evaluate(data_test, Y_test, verbose=0)
    st.success(f"🔍 Precisión del modelo sobre el conjunto de prueba: **{acc:.4f}**")
    st.info(f"📉 Pérdida (loss) del modelo sobre el conjunto de prueba: **{loss:.4f}**")
except Exception as e:
    st.warning(f"⚠️ No se pudo evaluar el modelo: {e}")