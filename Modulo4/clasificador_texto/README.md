# Clasificador de Texto con Deep Learning

Esta aplicación usa un modelo de redes neuronales convolucionales para clasificar textos en categorías predefinidas, entrenado con TensorFlow/Keras.

## 🔧 Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## 🚀 Cómo usar

1. Colocá tu archivo `df_total.csv` dentro de la carpeta `data/`.
2. Asegurate de tener el modelo entrenado (`modelo21.keras`) y su historial (`historial_entrenamiento.pkl`).
3. Ejecutá la app con:

```bash
streamlit run app.py
```

## 🧠 Qué incluye

- Clasificación de texto manual o por archivo `.txt`
- Gráfico de barras con probabilidades por clase
- Gráficas de pérdida y precisión por época
- Matriz de confusión
- Métrica final de precisión sobre el conjunto de prueba
