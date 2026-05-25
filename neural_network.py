# neural_network.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model():
    model = Sequential()
    # Corrected way to pass input shape in Keras 3
    model.add(Input(shape=(29,))) 
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam', 
        metrics=['accuracy']
    )
    return model
