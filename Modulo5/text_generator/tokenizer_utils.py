import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Limpiar el texto (opcional según tus necesidades)
def limpiar_texto(texto):
    """
    Limpia el texto, eliminando caracteres no deseados.
    """
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'\d+', '', texto)  # Eliminar números
    texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar puntuaciones
    return texto

# Crear el tokenizador y entrenarlo
def crear_tokenizador(textos, num_palabras=10000, max_len=100):
    """
    Crea un tokenizador para los textos.
    - `textos`: lista de textos a tokenizar.
    - `num_palabras`: número máximo de palabras que el tokenizador debe considerar.
    - `max_len`: longitud máxima de secuencia.
    """
    tokenizador = Tokenizer(num_words=num_palabras, oov_token="<OOV>")
    tokenizador.fit_on_texts(textos)
    secuencias = tokenizador.texts_to_sequences(textos)
    secuencias_pad = pad_sequences(secuencias, maxlen=max_len, padding='post', truncating='post')
    return tokenizador, secuencias_pad

# Guardar el tokenizador en un archivo .pkl
def guardar_tokenizador(tokenizador, ruta="tokenizer.pkl"):
    """
    Guarda el tokenizador en un archivo .pkl para su reutilización.
    """
    with open(ruta, 'wb') as f:
        pickle.dump(tokenizador, f)

# Cargar el tokenizador desde un archivo .pkl
def cargar_tokenizador(ruta="tokenizer.pkl"):
    """
    Carga el tokenizador desde un archivo .pkl.
    """
    with open(ruta, 'rb') as f:
        tokenizador = pickle.load(f)
    return tokenizador

# Preprocesar y tokenizar un solo texto
def tokenizar_texto(texto, tokenizador, max_len=100):
    """
    Tokeniza un solo texto, aplicando preprocesamiento y padding.
    """
    texto_limpio = limpiar_texto(texto)
    secuencia = tokenizador.texts_to_sequences([texto_limpio])
    secuencia_pad = pad_sequences(secuencia, maxlen=max_len, padding='post', truncating='post')
    return secuencia_pad
