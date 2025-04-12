import tensorflow as tf
from tensorflow import keras
from keras import layers

class CNNBiLSTMAMModel(tf.keras.Model):
    def __init__(self, time_steps=5, num_features=8, cnn_filters=64, bilstm_units=64, use_attention=True):
        super(CNNBiLSTMAMModel, self).__init__()

        self.use_attention = use_attention

        # CNN layer
        self.conv1d = layers.Conv1D(filters=cnn_filters, kernel_size=1, activation='relu', padding='same')

        # BiLSTM layer
        self.bilstm = layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True))

        # Attention layer (optional)
        if use_attention:
            self.att_dense = layers.Dense(1, activation='tanh')
            self.att_softmax = layers.Softmax(axis=1)

        # output layer
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.conv1d(inputs)                  # shape: (batch, time_steps, cnn_filters)
        x = self.bilstm(x)                       # shape: (batch, time_steps, bilstm_units*2)

        if self.use_attention:
            # Attention weight calculation
            e = self.att_dense(x)                # shape: (batch, time_steps, 1)
            alpha = self.att_softmax(e)          # shape: (batch, time_steps, 1)
            context = tf.reduce_sum(alpha * x, axis=1)  # shape: (batch, bilstm_units*2)
        else:
            context = tf.reduce_mean(x, axis=1)  # if no attention, use mean pooling

        output = self.output_layer(context)      # shape: (batch, 1)
        return output
