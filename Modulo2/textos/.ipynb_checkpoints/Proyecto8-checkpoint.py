#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;">
#     <h1 style="color: blue;">Anexo 8</h1>
#     <h3>Proyecto 8: Clasificador de texto</h3>
#     <hr/>
#     <p style="text-align: right;">Mg. Luis Felipe Bustamante Narváez</p>
# </div>

# En este ejercicio realizaremos un clasificador de texto basado en la forma escritural, sintáctica y semántica de dos escritores latinoamericanos, por un lado al argentino Jorge Luis Borges, y por otro frente al uruguayo Mario Benedetti.
# 
# Las principales diferencias entre sus trabajos:
# 
# 📖 1. Temas y Filosofía
# Borges: Su poesía es filosófica, abstracta y llena de referencias literarias, mitológicas y metafísicas. Le interesaban temas como el tiempo, el infinito, el destino, la identidad y la memoria. Su tono es intelectual y a veces enigmático.
# 
# Benedetti: Escribe de manera más directa y accesible. Sus temas son el amor, la vida cotidiana, la lucha social, el exilio y la esperanza. Su tono es cálido, humano y cercano al lector.
# 
# 🖋 2. Lenguaje y Estilo
# Borges: Usa un lenguaje elegante, erudito y con muchas metáforas complejas. Su poesía es reflexiva, con estructuras clásicas y a veces con formas fijas como sonetos.
# 
# Benedetti: Usa un lenguaje sencillo, directo y coloquial. Sus poemas parecen conversaciones o pensamientos escritos sin mucha ornamentación.
# 
# 🔄 3. Estructura y Ritmo
# Borges: Tiende a usar estructuras más tradicionales con rima y métrica cuidadas, aunque también experimenta con versos libres.
# 
# Benedetti: Prefiere el verso libre y la naturalidad del habla cotidiana, sin preocuparse demasiado por la métrica.

# ## Librerías

# In[ ]:


pip install tensorflow


# In[252]:


import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import PyPDF2
from IPython.display import display, HTML
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ## Cargamos los documentos

# In[11]:


def extraer_texto_desde_pdf(ruta_archivo):
    with open(ruta_archivo, 'rb') as archivo:
        lector = PyPDF2.PdfReader(archivo)
        texto = ''
        for pagina in range(len(lector.pages)):
            texto += lector.pages[pagina].extract_text()
        return texto


# In[12]:


ruta_carpeta = 'textos'


# ## Guardamos los textos en una lista

# In[13]:


todos_los_textos = []
for archivo in tqdm(os.listdir(ruta_carpeta)):
    if archivo.endswith('.pdf'):
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        try:
            documento = extraer_texto_desde_pdf(ruta_completa)
            todos_los_textos.append(documento)
        except Exception as e:
            print(f'Error al procesar {archivo}: {e}')


# In[ ]:


todos_los_textos[0]


# ## Procesamiento de los datos

# Vamos a separar los textos por etiquetas, enumerando los textos de Borgues con la etiqueta 0 y los de Benedetti con la etiqueta 1

# In[ ]:


# Eliminamos espacios al inicio y al final para evitar problemas con el pdf
for texto in todos_los_textos:
    archivo = texto.strip()
archivo


# In[15]:


# Creamos las listas vacías
textos = []
etiquetas = []


# In[ ]:


# Mostramos línea por línea los textos de cada escritor
#count = 0   #contador de linea para pruebas
for etiqueta, texto in enumerate(todos_los_textos):
    print(f'\n--- Texto {etiqueta} ---\n')
    for linea in texto.split(' \n'):
        #count += 1   #contador de líneas para prueba
        print(linea)
        # Convertimos a minúsculas
        linea = linea.rstrip().lower()
        print(linea)
        # Eliminamos signos de puntuación
        if linea:
            linea = linea.translate(str.maketrans('','', string.punctuation))
            print(linea)
            # Agregamos el texto límpio y le asignamos su respectiva etiqueta
            textos.append(linea)
            etiquetas.append(etiqueta)
    


# In[ ]:


# Mostramos las listas
textos


# ## Entrenamiento

# X representa la lista de los textos, y Y, representa la lista de las etiquetas, quien sería nuestra variable a predecir.

# In[72]:


X_train, X_test, Y_train, Y_test = train_test_split(textos, etiquetas, train_size=0.9, random_state=42)


# In[73]:


# Mostramos en forma de tupla el tamaño de cada muestra
len(Y_train), len(Y_test)


# In[76]:


# Probamos las muestras de entrenamiento
X_train[0], Y_train[0]


# In[77]:


# Probamos las muestras de prueba
X_test[0], Y_test[0]


# ### Representación de palabras desconocidas

# #### <b>&lt;unk&gt;</b>
# 
# Es una convención utilizada a menudo en <b>NPL</b> para representar palabras desconocidas o fuera del vocabulario. Por ejemplo, <span style="color: fuchsia;">si una palabra no se encontró en la muestra de entrenamiento, pero aparece en la muestra de prueba, será desconocida, y se requiere agregarle un índice que diferencie a esta palabra</span>.
# 
# 

# In[78]:


indice = 1
indice_palabras = {'<unk>': 0}


# ## Construcción del diccionario de codificación de palabras a índice

# In[81]:


for texto in X_train:
    # Separamos cada palabra
    tokens = texto.split()
    #print(tokens) # Probamos como se ven los tokens
    for token in tokens:
        # Buscamos si la palabra no está en el índice para luego agregarla sin repetir
        if token not in indice_palabras:
            indice_palabras[token] = indice
            indice += 1


# In[ ]:


# Mostramos el índice de palabras - palabras únicas
indice_palabras


# In[ ]:


# tamaño de palabras únicas
indice_palabras


# ### Conversión del índice de palabras de String a enteros
# 
# Cómo el entrenamiento no se debe hacer con palabras, creamos una muestra convertida a su valor específico en enteros

# In[234]:


# listas para enteros
X_train_int = []
X_test_int = []
# Banderas para ejecutarse una sola vez
X_int_train_hecho = False
X_int_test_hecho = False


# In[ ]:


# Conversión de los datos de entrenamiento
if not X_int_train_hecho:
    for texto in X_train:
        # dividimos de nuevo en palabras
        tokens = texto.split()
        # Por cada palabra encontrada la cambia por su valor numérico de la clave del diccionario
        linea_entero = [ indice_palabras[token] for token in tokens ]
        #print(linea_entero)
        # Agregamos el nuevo valor a la lista de entrenamiento
        X_train_int.append(linea_entero)
    X_int_train_hecho = True
    print("Conversión de entrenamiento ejecutada con éxito.")
else:
    print("La conversión de entrenamiento ya se había ejecutado previamente.")
    


# In[ ]:


# Mostramos la conversión de entrenamiento
X_train_int


# In[237]:


# Conversión de los datos de prueba -- Como puede haber desconocidos, debemos hacer esto:
if not X_int_test_hecho:
    for texto in X_test:
        tokens = texto.split()
        linea_entero = [indice_palabras.get(token, 0) for token in tokens] #trae el token o 0
        #print(linea_entero)
        X_test_int.append(linea_entero)
    X_int_test_hecho = True
    print("Conversión de prueba ejecutada con éxito.")
else:
    print("La conversión de prueba ya se había ejecutado previamente.")
    


# In[223]:


# Mostramos la conversión de prueba
len(X_test_int)


# ## Matriz de Transición

# Como se indicó en la teoría de los procesos de <b style='color:blue'>Markov</b>, se requiere construir una matriz de transición y los estados iniciales para cada escritor:
# 
# 1. Creamos un vector <b style='color:fuchsia;'>V</b> con el tamaño total del <b style='color:fuchsia;'>indice_palabras</b>
# 2. Creamos la matriz <b style='color:fuchsia;'>A0</b> para las palabras de <b style='color:blue;'>Borges</b>
# 3. Creamos el vector de probabilidad inicial <b style='color:fuchsia;'>pi0</b> para las palabras de <b style='color:blue;'>Borges</b>
# 4. Creamos la matriz <b style='color:fuchsia;'>A1</b> para las palabras de <b style='color:blue;'>Benedeti</b>
# 5. Creamos el vector de probabilidad inicial <b style='color:fuchsia;'>pi1</b> para las palabras de <b style='color:blue;'>Benedeti</b>
# 

# In[102]:


V = len(indice_palabras)
# Creamos las matrices y vectores con 1 para poder hacer el suavizado
A0 = np.ones((V, V)) 
pi0 = np.ones(V)
A1 = np.ones((V, V)) 
pi1 = np.ones(V)
#Motramos, por ejemplo
pi0


# ## Función de conteo de palabras

# In[103]:


def compute_counts(texto_as_int, A, pi):
    #Recorremos los tokens
    for tokens in texto_as_int:
        #Creamos el posible último elemento como referencia
        last_index = None
        #Recorremos cada elemento de cada línea
        for index in tokens:
            #Nos ubicamos en la primera secuencia
            if last_index is None:
                # Agregamos el valor inicial
                pi[index] +=1
            else:
                # Agregamos los valores a la matriz
                A[last_index, index] += 1
            # Asignamos el valor actual al last_index
            last_index = index


# In[104]:


# Llamamos la función
#Para Borges
compute_counts([t for t, y in zip(X_train_int, Y_train) if y == 0], A0, pi0)
#Para Benedetti
compute_counts([t for t, y in zip(X_train_int, Y_train) if y == 1], A1, pi1)


# In[114]:


# Probamos
A1


# ### Explicación
# <b style='color:blue;'>
# pi0 = array([ 1., 10.,  1., ...,  1.,  1.,  1.])
# </b>
# <hr>
# <b style='color:red;'>
# pi1 = array([ 1., 14.,  1., ...,  1.,  1.,  1.])
# </b>
# <hr>
# <b style='color:orange;'>
# A0 = array([[1., 1., 1., ..., 1., 1., 1.],
#        [1., 1., 1., ..., 1., 1., 1.],
#        [1., 1., 1., ..., 1., 1., 1.],
#        ...,
#        [1., 1., 1., ..., 1., 1., 1.],
#        [1., 1., 1., ..., 1., 1., 1.],
#        [1., 1., 1., ..., 1., 1., 1.]])
# </b>
# <hr>
# <b style='color:green;'>
# A1 =  array([[1., 1., 1., ..., 1., 1., 1.],
#        [1., 1., 2., ..., 1., 1., 1.],
#        [1., 1., 1., ..., 1., 1., 1.],
#        ...,
#        [1., 1., 1., ..., 1., 1., 1.],
#        [1., 1., 1., ..., 1., 1., 1.],
#        [1., 1., 1., ..., 1., 1., 1.]])
# </b>
# <hr>
# En el vector inicial <b style='color:blue;'>pi0</b>, la primera posición corresponde a los unk, la segunda posición corresponde a la palabra <b>o</b>, y nos está indicando que en los textos de Borges aparece iniciando la línea 10 veces. Si observamos los textos de Benedetti, <b style='color:red;'>pi1</b> , nos indica que aparece 14 veces comenzando la línea. De esta manera podemos proceder a encontrar la probabilidad.
# 
# Con respecto a las matrices de transición, podemos observar en la matriz de Benedetti, <b style='color:green;'>A1</b>, que en la segunda fila, hubo una transición de la palabra actual a la siguiente, valores que nos indicarán el comportamiento natural de los textos.

# ## Distribución de Probabilidad

# <b style='color:red;'>Normalizamos</b>
# 
# Para observar las probabilidades, se requiere normalizar los vectores y matrices generados en el conteo, para que su valor oscile entre 0 y 1, como debe ser.
# 
# Esta es una manera empírica de demostrar las fórmulas mencionadas en la teoría

# In[117]:


#Conservamos los datos originales A0, A1, pi0 y pi1, creando las variables nomralizadas
#Para esto vamos a guardar los datos originales
A0_norm = A0.copy()
pi0_norm = pi0.copy()
A1_norm = A1.copy()
pi1_norm = pi1.copy()
# Bandera de normalizado
normalize = False


# In[121]:


# Borges
if not normalize:
    # Borges
    A0_norm /= A0_norm.sum(axis=1, keepdims = True)
    pi0_norm /= pi0_norm.sum()
    # Benedetti
    A1_norm /= A1_norm.sum(axis=1, keepdims = True)
    pi1_norm /= pi1_norm.sum()
    print("Normalización ejecutada con éxito.")
    normalize = True
else:
    print("Las variables ya fueron normalizadas previamente.")


# In[122]:


# Probamos
pi0_norm


# In[123]:


A0_norm


# ### Explicación
# <b style='color:blue;'>
# pi0_norm = array([0.00023111, 0.00231107, 0.00023111, ..., 0.00023111, 0.00023111,
#        0.00023111])
# </b>
# <hr>
# 
# <b style='color:red;'>
# A0_norm = array([[0.00029472, 0.00029472, 0.00029472, ..., 0.00029472, 0.00029472,
#         0.00029472],
#        [0.00029155, 0.00029155, 0.00029155, ..., 0.00029155, 0.00029155,
#         0.00029155],
#        [0.00029455, 0.00029455, 0.00029455, ..., 0.00029455, 0.00029455,
#         0.00029455],
#        ...,
#        [0.00029472, 0.00029472, 0.00029472, ..., 0.00029472, 0.00029472,
#         0.00029472],
#        [0.00029472, 0.00029472, 0.00029472, ..., 0.00029472, 0.00029472,
#         0.00029472],
#        [0.00029464, 0.00029464, 0.00029464, ..., 0.00029464, 0.00029464,
#         0.00029464]])
# </b>
# <hr>
# 
# En el vector inicial <b style='color:blue;'>pi0</b>, observemos que no aparece ningún valor en cero, lo que indica que el método de suavizar funciónó perfectamente, al igual que en la matriz <b style='color:red;'>A0</b>, lo que permite una <b>distribución de probabilidad</b>.

# ## Espacio logarítmico

# Como vimos en la teoría, estas probabilidades pueden tener un desbordamiento por debajo, ya que se aproximan a cero, entonces, para evitar errores computacionales, usaremos el espacio logarítmico.

# In[124]:


#Borges
log_A0_norm = np.log(A0_norm)
log_pi0_norm = np.log(pi0_norm)
#Benedetti
log_A1_norm = np.log(A1_norm)
log_pi1_norm = np.log(pi1_norm)


# In[125]:


# Probamos
log_pi0_norm


# In[126]:


# Probamos
log_A0_norm


# ## Pre-análisis

# Vamos a revisar diferentes elementos que nos permitan entender mejor lo que desarrollamos

# In[161]:


# Conteo de etiquetas de clase 0 (Borges) en Y_train
count_Y_0 = sum(y == 0 for y in Y_train)
# Conteo de etiquetas de clase 1 (Benedetti) en Y_train
count_Y_1 = sum(y == 1 for y in Y_train)
# Cantidad total de ejemplos de entrenamiento
total = len(Y_train)
# Probabilidad a priori de la clase 0
p0 = count_Y_0 / total
# Probabilidad a priori de la clase 1
p1 = count_Y_1 / total
# Logaritmo de la clase a priori 0
log_p0 = np.log(p0)
# Logaritmo de la clase a priori 0
log_p1 = np.log(p1)

display(HTML(f'''
Se encontró {count_Y_0} etiquetas de clase 0, <b style='color:fuchsia;'>Borges</b>,<br>
Se encontró {count_Y_1} etiquetas de clase 1, <b style='color:skyblue;'>Benedetti</b>,<br>
para un total de <b style='color:red;'>{total}</b> ejemplos de entrenamiento.
<hr>
Las probabilidades a priori serían las siguientes:<br>
<table style="border: 1px solid black; border-collapse: collapse;">
  <tr>
    <td style="border: 1px solid black; padding: 5px;">Borges</td>
    <td style="border: 1px solid black; padding: 5px;">{p0}</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 5px;">Benedetti</td>
    <td style="border: 1px solid black; padding: 5px;">{p1}</td>
  </tr>
</table>
<hr>
Como usamos el espacio logarítmico, estas serían las probabilidades reales de encontrar un texto de la clase 0 o 1:
<table style="border: 1px solid black; border-collapse: collapse;">
  <tr>
    <td style="border: 1px solid black; padding: 5px;">Borges</td>
    <td style="border: 1px solid black; padding: 5px;">{log_p0}</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 5px;">Benedetti</td>
    <td style="border: 1px solid black; padding: 5px;">{log_p1}</td>
  </tr>
</table>
<hr>
'''))


# ## Construcción del Clasificador

# In[165]:


# Creamos una clase
class Classifier:
    # Constructor
    def __init__(self, log_As, log_pis, log_apriors):
        self.log_As = log_As
        self.log_pis = log_pis
        self.log_apriors = log_apriors
        # número de clases
        self.k = len(log_apriors)

    # Método de verosimilitud
    def _compute_log_likelihood(self, input_, class_):
        log_A = self.log_As[class_]
        log_pi = self.log_pis[class_]
        #Repetimos lo hecho en el ejemplos de creación de la matriz
        last_index = None
        log_prob = 0
        #Recorremos la entrada del usuario
        for index in input_:
            if last_index is None:
                #Primer token en la secuencia
                log_prob += log_pi[index]
            else:
                #Calculamos la probabilidad de transición del a palabra anterior a la actual
                log_prob += log_A[last_index, index]
            #Actualizamos el index para la próxima iteración
            last_index = index
        return log_prob

    # Función de predicción
    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            # Calcula los logaritmos de las probabilidades posteriores para cada clase
            posteriors = [self._compute_log_likelihood(input_, c) + self.log_apriors[c] \
                          for c in range(self.k)]
            #Elige la clase de mayor probabilidad posterior como la predicción
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions


# ### Explicación

# 1️⃣ Constructor (__init__)
# 📌 ¿Qué parámetros recibe?
# 
# log_As: Matrices de probabilidades de transición entre palabras en logaritmo.
# log_pis: Probabilidades iniciales de cada palabra en logaritmo.
# log_apriors: Probabilidades previas (prior) de cada clase en logaritmo.
# self.k: Número total de clases.
# 
# 📌 ¿Por qué usa logaritmos?
# ✅ Evita problemas de underflow cuando se multiplican muchas probabilidades pequeñas.
# ✅ Convierte productos en sumas, lo que hace más fácil la optimización.
# 
# 2️⃣ Método _compute_log_likelihood (Cálculo de Verosimilitud)
# 
# 📌 ¿Qué hace?
# 
# Calcula la log-verosimilitud de una secuencia (input_) dada una clase (class_).
# Usa la probabilidad inicial de la primera palabra (log_pi[index]).
# Luego, suma las probabilidades de transición entre palabras (log_A[last_index, index]).
# Retorna log_prob, que indica qué tan probable es la secuencia dada la clase.
# 
# 3️⃣ Método predict (Clasificación)
# 
# 📌 ¿Qué hace?
# 
# Inicializa predictions con ceros (un array para almacenar las predicciones).
# 
# Para cada entrada en inputs:
# Calcula las log-verosimilitudes para cada clase.
# Suma la probabilidad previa (prior) log_apriors[c] de cada clase.
# Elige la clase con mayor probabilidad posterior usando np.argmax().
# Devuelve predictions, que contiene las clases predichas.
# 

# ## Objeto de la clase Clasifier

# In[167]:


# Creamos un objeto de la clase Clasifier para llamar los métodos del clasificador
clf = Classifier([log_A0_norm, log_A1_norm], [log_pi0_norm, log_pi1_norm], [log_p0, log_p1])


# ### Explicación
# 
# La clase Classifier, recibe 3 parámetros en su constructor, es decir 3 atributos. En su orden estos atributos son:
# 
# 1️⃣[log_A0_norm, log_A1_norm] que serán los argumentos de log_As
# 
# 2️⃣[log_pi0_norm, log_pi1_norm] que serán los argumentos de log_pis
# 
# 3️⃣[log_p0, log_p1] que serán los argumentos de log_apriors
# 
# Es decir,
# 
# 1️⃣ Las matrices de transición normalizadas
# 
# 2️⃣ Los vectores con los valores iniciales o estados iniciales
# 
# 3️⃣ Las probabilidades de cada clase utilizando el espacio logarítmico
# 

# ## Predicción

# In[228]:


# Llamamos al método predict  (Datos de entrenamiento: Aprox 1.0)
P_train = clf.predict(X_train_int)
# Mostramos la predicción con la muestra de entrenamiento
print(f'Accuraci Train: {np.mean(P_train == Y_train)}')


# In[201]:


len(X_test_int)


# In[229]:


# Llamamos al método predict (Datos de prueba)
P_test = clf.predict(X_test_int)
# Mostramos la predicción con la muestra de prueba
print(f'Accuraci Test: {np.mean(P_test == Y_test)}')


# ## Conclusiones

# <div style="text-align: center;">
#     <p>Se puede observar como los datos de prueba, permiten clasificar los textos con un 75% de presición, de tal manera que al poner textos de estos poetas, fácilmente podrá indicar quien lo escribió.</p>
#     <hr/>
#     <p style="text-align: right;">Mg. Luis Felipe Bustamante Narváez</p>
# </div>
