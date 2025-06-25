from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, Dense, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def crear_modelo_rnn(input_length, num_clases, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=input_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_clases, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def crear_modelo_cnn(input_length, num_clases, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=input_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_clases, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def crear_modelo_transformer(input_length, num_clases, embedding_dim=100, num_heads=2, ff_dim=128):
    # Modelo Transformer básico
    inputs = Input(shape=(input_length,))
    x = Embedding(input_dim=10000, output_dim=embedding_dim)(inputs)
    
    # Capa Transformer: MultiHeadAttention y Normalización
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
    x = LayerNormalization()(attention)
    x = Dropout(0.2)(x)  # Añadir Dropout para evitar sobreajuste
    
    # Agregar una capa de pooling global
    x = GlobalMaxPooling1D()(x)
    
    # Capa densa para clasificación
    x = Dense(64, activation='relu')(x)
    output = Dense(num_clases, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
