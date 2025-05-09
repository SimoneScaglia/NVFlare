import tensorflow as tf
from tensorflow import keras

class CNN(tf.keras.Model):
    def __init__(self, input_dim=40):
        super().__init__()
        self._input_shape = (None, input_dim)  # Aggiungi questo attributo
        self.model = get_net(input_dim)

    def call(self, inputs):
        return self.model(inputs)

def get_net(input_shape):
    return keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_shape,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

def get_opt():
    return tf.keras.optimizers.Adam(learning_rate=0.001)

def get_metrics():
    return [tf.keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=1000)]