import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tokenizer_utils import crear_tokenizador, guardar_tokenizador
from modelo_factory import crear_modelo_rnn, crear_modelo_cnn, crear_modelo_transformer

def entrenar_modelo(textos, etiquetas, modelo_tipo="rnn", num_clases=3, max_len=100):
    # Preprocesamiento y tokenización
    tokenizador, secuencias_pad = crear_tokenizador(textos, max_len=max_len)
    guardar_tokenizador(tokenizador)

    # Convertir etiquetas a formato categórico
    etiquetas = to_categorical(etiquetas, num_classes=num_clases)
    
    # Dividir en conjunto de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(secuencias_pad, etiquetas, test_size=0.2, random_state=42)

    # Crear el modelo
    if modelo_tipo == "rnn":
        modelo = crear_modelo_rnn(input_length=max_len, num_clases=num_clases)
    elif modelo_tipo == "cnn":
        modelo = crear_modelo_cnn(input_length=max_len, num_clases=num_clases)
    elif modelo_tipo == "transformer":
        modelo = crear_modelo_transformer(input_length=max_len, num_clases=num_clases)
    else:
        raise ValueError("Modelo no soportado. Elige entre 'rnn', 'cnn' o 'transformer'.")

    # Entrenamiento
    modelo.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

    # Guardar el modelo entrenado
    modelo.save(f"modelos/modelo_{modelo_tipo}.keras")
    return modelo
