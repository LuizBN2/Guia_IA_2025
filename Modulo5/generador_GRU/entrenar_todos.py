import streamlit as st
import os
import psutil
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Input, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import json
from streamlit_utils import StreamlitProgressCallback  # Importa el callback


# ============================
# CONFIGURACIÓN
# ============================
MAX_VOCAB_SIZE = 5000
EMBEDDING_DIM = 100
UNITS = 150
EPOCHS = 20
NUM_HEADS = 2
FF_DIM = 128
MAXLEN_BASE = 50  # longitud inicial de las secuencias

CORPUS_PATH = "corpus_completo.txt"
TOKENIZER_PATH = "tokenizer.pkl"
MODELOS_DIR = "modelos"
HISTORIALES_DIR = "historiales"

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(HISTORIALES_DIR, exist_ok=True)

# ============================
# 1. VERIFICACIÓN DE MEMORIA DISPONIBLE
# ============================
def verificar_memoria():
    memoria_disponible = psutil.virtual_memory().available / (1024 ** 3)  # en GB
    print(f"Memoria disponible: {memoria_disponible:.2f} GB")
    return memoria_disponible

# ============================
# 2. PREPROCESAMIENTO
# ============================
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    texto = f.read().lower()

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts([texto])
total_words = len(tokenizer.word_index) + 1

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

tokens = tokenizer.texts_to_sequences([texto])[0]

# ============================
# 3. AJUSTE DINÁMICO DE LONGITUD DE SECUENCIAS
# ============================
def ajustar_longitud_secuencias(memoria_disponible):
    if memoria_disponible > 2:  # Si hay más de 2GB de memoria disponible, podemos usar más tokens
        return 100
    elif memoria_disponible > 1:  # Si hay más de 1GB, usamos 75 tokens
        return 75
    else:
        return MAXLEN_BASE  # Usamos 50 si la memoria es limitada

# Ajustamos la longitud según la memoria disponible
memoria_disponible = verificar_memoria()
max_seq_len = ajustar_longitud_secuencias(memoria_disponible)
print(f"Usando longitud de secuencias: {max_seq_len} tokens\n")

# ============================
# 4. FUNCIONES DE CREACIÓN DE SECUENCIAS
# ============================
if 'st' in globals():  # para no romper si lo ejecutas fuera de Streamlit
    st.write("🔄 Generando secuencias...")
    barra = st.progress(0)
else:
    barra = None

sequences = []
for i in range(1, len(tokens)):
    n_gram_sequence = tokens[:i+1]
    if len(n_gram_sequence) <= max_seq_len:
        sequences.append(n_gram_sequence)
    if barra:
        barra.progress(i / len(tokens))

sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

X = sequences[:, :-1]
y = to_categorical(sequences[:, -1], num_classes=total_words)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# ============================
# 5. GUARDAR PROGRESO JSON
# ============================
def guardar_progreso_json(epoch, total_epochs, loss, accuracy):
    progreso = {
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        "loss": float(loss),
        "accuracy": float(accuracy)
    }
    os.makedirs(HISTORIALES_DIR, exist_ok=True)
    with open(os.path.join(HISTORIALES_DIR, "progreso_entrenamiento.json"), "w") as f:
        json.dump(progreso, f)



# ============================
# 5. FUNCIONES DE PROGRESO
# ============================
import json

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, container, total_epochs):
        super().__init__()
        self.container = container
        self.total_epochs = total_epochs
        self.progress_bar = None
        self.status_text = None
        self.epoch = 0

    def on_train_begin(self, logs=None):
        self.progress_bar = self.container.progress(0)
        self.status_text = self.container.empty()
        self.epoch = 0
        self.status_text.text("⏳ Entrenamiento iniciado...")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)

        # Actualizar el progreso en el archivo JSON
        progreso = {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "accuracy": acc,
            "val_accuracy": val_acc,
            "loss": loss,
            "val_loss": val_loss
        }
        with open("historiales/progreso_entrenamiento.json", "w") as f:
            json.dump(progreso, f)

        self.status_text.markdown(
            f"📈 **Época {self.epoch}/{self.total_epochs}**  \n"
            f"✅ Accuracy: `{acc:.4f}` | 🔍 Val_Accuracy: `{val_acc:.4f}`  \n"
            f"❌ Loss: `{loss:.4f}` | 🔎 Val_Loss: `{val_loss:.4f}`"
        )
        self.progress_bar.progress(int((self.epoch / self.total_epochs) * 100))

    def on_train_end(self, logs=None):
        self.status_text.text("🏁 Entrenamiento finalizado.")
        self.progress_bar.progress(100)



# ============================
# 6. CALLBACK SIN STREAMLIT
# ============================
class JsonProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0)
        acc = logs.get("accuracy", 0)
        guardar_progreso_json(epoch, self.total_epochs, loss, acc)



# ============================
# 6. ENTRENAMIENTO DEL MODELO
# ============================
def entrenar_modelo(tipo, st_container=None):
    print(f"\n🔁 Entrenando modelo {tipo.upper()}...")

    model = Sequential()
    model.add(Embedding(total_words, EMBEDDING_DIM, input_length=max_seq_len-1))

    if tipo == "gru":
        model.add(GRU(UNITS))
    elif tipo == "lstm":
        model.add(LSTM(UNITS))
    else:
        raise ValueError("Solo se admiten GRU o LSTM")

    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = []
    if st_container:
        callbacks.append(StreamlitProgressCallback(st_container, EPOCHS))
    else:
        callbacks.append(JsonProgressCallback(EPOCHS))

    # Guardar archivo de progreso inicial (época 0)
    guardar_progreso_json(0, EPOCHS, 0, 0)


    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        verbose=0,
        callbacks=callbacks
    )

    model.save(f"{MODELOS_DIR}/modelo_{tipo}.keras")
    pd.DataFrame(history.history).to_csv(f"{HISTORIALES_DIR}/historial_{tipo}.csv", index=False)

    print(f"✅ Modelo {tipo.upper()} guardado.\n")


# ============================
# 7. ENTRENAMIENTO DEL MODELO TRANSFORMER
# ============================
def transformer_block(inputs):
    attention_output = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBEDDING_DIM)(inputs, inputs)
    attention_output = Dropout(0.1)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn_output = Dense(FF_DIM, activation="relu")(out1)
    ffn_output = Dense(EMBEDDING_DIM)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)

    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def entrenar_transformer(st_container=None):
    print("\n🚀 Entrenando modelo TRANSFORMER...")

    inputs = Input(shape=(max_seq_len-1,))
    x = Embedding(total_words, EMBEDDING_DIM)(inputs)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(UNITS, activation="relu")(x)
    outputs = Dense(total_words, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Mixed precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    callbacks = []
    if st_container:
        callbacks.append(StreamlitProgressCallback(st_container, EPOCHS))
    else:
        callbacks.append(JsonProgressCallback(EPOCHS))

    guardar_progreso_json(0, EPOCHS, 0, 0)


    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        verbose=0,
        callbacks=callbacks
    )

    model.save(f"{MODELOS_DIR}/modelo_transformer.keras")
    pd.DataFrame(history.history).to_csv(f"{HISTORIALES_DIR}/historial_transformer.csv", index=False)

    print("✅ Modelo TRANSFORMER guardado.\n")


# ============================
# 8. EJECUCIÓN DEL PROCESO DE ENTRENAMIENTO
# ============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("⚠️ Debes indicar el modelo a entrenar: gru, lstm o transformer")
        sys.exit(1)

    tipo_modelo = sys.argv[1].lower()

    # Si hay argumento de número de épocas, lo usamos
    if len(sys.argv) >= 3:
        try:
            EPOCHS = int(sys.argv[2])
        except ValueError:
            print("⚠️ El número de épocas debe ser un número entero.")
            sys.exit(1)

    if tipo_modelo == "gru":
        entrenar_modelo("gru")
    elif tipo_modelo == "lstm":
        entrenar_modelo("lstm")
    elif tipo_modelo == "transformer":
        entrenar_transformer()
    else:
        print("⚠️ Modelo no reconocido. Usa: gru, lstm o transformer.")
