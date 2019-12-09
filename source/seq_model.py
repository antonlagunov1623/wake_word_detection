from keras import layers
from keras.models import Sequential
from keras import regularizers

def build():
    model = Sequential()
    model.add(layers.LSTM(64,input_shape=(23, 39), return_sequences=True,   kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.4))
    model.add(layers.LSTM(128, return_sequences=True,   kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.4))
    model.add(layers.LSTM(256, return_sequences=True,   kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu',   kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu',   kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu',   kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu',   kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model