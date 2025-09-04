import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
tf.keras.utils.set_random_seed(42)

class FCN(tf.keras.Model):
    def __init__(self, input_dim=40):
        super().__init__()
        self._input_shape = (None, input_dim)  # Aggiungi questo attributo
        self.model = get_net(input_dim)

    def call(self, inputs):
        return self.model(inputs)

def get_net(input_shape):
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=42)
    bias_initializer = tf.keras.initializers.Zeros()
    
    return keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_shape,), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    ])

def get_opt():
    return tf.keras.optimizers.Adam(learning_rate=0.001)

def get_metrics():
    return [
        AUC(name='auc', curve='ROC', num_thresholds=1000),
        AUC(name='auprc', curve='PR', num_thresholds=1000),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall')
    ]