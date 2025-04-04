import tensorflow as tf
from tensorflow.keras import layers, models

class CNN(tf.keras.Model):
    def __init__(self, input_dim=40):
        super().__init__()
        self._input_shape = (None, input_dim)  # Aggiungi questo attributo
        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)
