#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;">
#     <h1 style="color: #a64dff;">Anexo 23</h1>
#     <h3>Proyecto 23: Clasificador de Texto con RNN's para NLP</h3>
#     <hr/>
#     <p style="text-align: right;">Mg. Luis Felipe Bustamante Narváez</p>
# </div>

# En este proyecto, diseñaremos un clasificador de texto, utilizando redes neuronales recurrentes, recurso que permite un procesamiento más complejo de los ejercicios anteriores, pero a su vez más preciso y con excelente eficiencia, y mínimo coste computacional.

# ## Librerías

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import itertools
from keras.models import load_model
from keras.optimizers import RMSprop
import pickle


# ## Cargamos los datos

# In[2]:


path = 'data/df_total.csv'
df = pd.read_csv(path)


# In[3]:


df


# ## Procesamiento de Datos

# ### Creamos las categorías

# In[4]:


target = df['Type'].astype('category').cat.codes


# In[5]:


target


# In[6]:


# Adicionamos la columna al df
df['target'] = target


# In[7]:


df


# ### Separamos los conjuntos de Datos

# In[8]:


df_train, df_test = train_test_split(df, test_size=0.3)


# ### Obtenemos el número de clases

# In[9]:


K = df['target'].max() + 1
K


# ### Creamos los conjuntos de salida

# In[10]:


Y_train = df_train['target']
Y_test = df_test['target']


# ## Tokenización

# ### Tokenizamos oraciones en secuencias

# In[11]:


#Vocabulario máximo
max_vocab_size = 30000
#Iniciamos el tokenizador
tokenizer = Tokenizer(num_words=max_vocab_size)
#Tokenizamos
tokenizer.fit_on_texts(df_train['news'])
#Creamos las secuencias
secuences_train = tokenizer.texts_to_sequences(df_train['news'])
secuences_test = tokenizer.texts_to_sequences(df_test['news'])


# ### Diccionario de palabras tokenizadas

# In[12]:


# Creamos el diccionario
word2index = tokenizer.word_index
# Calculamos el tamaño del tokenizado
V = len(word2index)
# mostramos
print(f'Se encontraron {V} tokens.')


# In[13]:


diez = dict(itertools.islice(word2index.items(), 10))
print(f'Estas son las 10 primeras palabras que más se repiten son:\n{diez}')


# ### Rellenamos las Secuencias (padding)

# In[14]:


# Rellenar la secuencia de entrenamiento
data_train = pad_sequences(secuences_train)
print(f'Dimensiones del tensor de entrenamiento: {data_train.shape}')
# Longitud de la secuencia de entrenamiento
T = data_train.shape[1]
print(f'Longitud de la secuencia de entrenamiento: {T}')


# In[15]:


# Rellenar la secuencia de prueba
data_test = pad_sequences(secuences_test, maxlen=T)
print(f'Dimensiones del tensor de prueba: {data_test.shape}')
# Longitud de la secuencia de prueba
print(f'Longitud de la secuencia de prueba: {data_test.shape[1]}')


# ## Embedding y Modelo

# ### Dimensiones del Embedding

# In[16]:


D = 20


# ### Construcción del Modelo

# In[17]:


# Capa de entrada
i = Input(shape=(T,))
# Capa de embedding
x = Embedding(V + 1, D)(i) #+1 para el token especial de palabras desconocidas padding
# Capa de convolución
x = LSTM(32, return_sequences=True)(x)  # 32 filtros para las secuencias de palabras
# Capa de pooling
x = GlobalMaxPooling1D()(x)
# Capa Densa
x = Dense(K)(x)
# Creación del modelo
modelo = Model(i, x)


# ### Resumen del Modelo

# In[18]:


modelo.summary()


# ### Compilamos el Modelo

# In[19]:


modelo.compile(
    loss= SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']    
)


# ### Entrenamos el Modelo

# In[ ]:


print('Entrenando el modelo...')
r = modelo.fit(
    data_train,
    Y_train,
    epochs=50,
    validation_data=(data_test, Y_test)
)


# ### Gráfico de la pérdida por iteración

# In[21]:


# Gráfico de la función de pérdida (loss)
plt.plot(r.history['loss'], label='Pérdida del Entrenamiento', color='blue')
plt.plot(r.history['val_loss'], label='Pérdida del set de prueba', color='orange')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Evolución del entrenamiento')
plt.legend()
plt.grid(True)
plt.show()


# ### Gráfico de la presición por iteración

# In[22]:


# Gráfico de la métrica de presición (accuracy)
plt.plot(r.history['accuracy'], label='Presición del Entrenamiento', color='red')
plt.plot(r.history['val_accuracy'], label='Presición del set de prueba', color='black')
plt.xlabel('Épocas')
plt.ylabel('Presición')
plt.title('Evolución del entrenamiento')
plt.legend()
plt.grid(True)
plt.show()


# ### Guardar Modelo y otros archivos necesarios

# In[23]:


# Archivo con extensión HDF5 (deprecado) o keras (actual)
modelo.save('modelo23.keras')
print('Modelo guardado con éxito.')


# In[24]:


# Guardamos los pesos
modelo.save_weights('modelo23_pesos.weights.h5')
print('Pesos del modelo guardados con éxito.')


# In[25]:


# Guardamos el historial del modelo
with open('historial_entrenamiento_23.pkl', 'wb') as f:
    pickle.dump(r.history, f)


# In[26]:


# Guardamos el tokenizador
with open('tokenizer_23.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


# ### Cargar el Modelo para futuras pruebas

# In[27]:


model_load = load_model('modelo23.keras', compile=False)

model_load.compile(
    optimizer = RMSprop(),
    loss = SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

print(f'El Modelo {model_load} se ha cargado, y recompilado correctamente.')


# ## Probamos el Modelo con Datos Nuevos

# ### Función para predecir texto

# In[28]:


def predecir_texto(texto, modelo, tokenizer, T, idx2label=None):
    # Asegurarse que el texto está en una lista
    if isinstance(texto, str):
        texto = [texto]

    # Tokenizar y hacer padding
    secuencia = tokenizer.texts_to_sequences(texto)
    secuencia_padded = pad_sequences(secuencia, maxlen=T)

    # Predicción
    pred = modelo.predict(secuencia_padded)
    clase_predicha = np.argmax(pred, axis=1)[0]

    # Mostrar resultado
    if idx2label:
        print(f'Clase predicha: {clase_predicha} ({idx2label[clase_predicha]})')
        return idx2label[clase_predicha]
    else:
        print(f'Clase predicha: {clase_predicha}')
        return clase_predicha


# ### Llamamos la función

# In[29]:


texto_de_prueba = "Este es un ejemplo de noticia económica internacional"
# Crear mapeo inverso de índices a nombres
idx2label = dict(enumerate(df['Type'].astype('category').cat.categories))
predecir_texto(texto_de_prueba, model_load, tokenizer, T)
idx2label


# ## Conclusiones

# <div style="text-align: center;">
#     <p>En este modelo, se logró crear una clasificación a través de keras, capaz de indetificar, a partir del contenido de una noticia, cuál es su cateoría. Con un entrenamiento a través de embeddings y redes neuronales recurrentes, hemos generado un clasificador de mayor presición capaz de clasificar cualquier texto informativo. Aunque el tiempo de entrenamiento es mucho mayor, la precisión del modelo es bastante mejor que los anteriores.</p>
#         <hr/>
#     <p style="text-align: right;">Mg. Luis Felipe Bustamante Narváez</p>
# </div>
