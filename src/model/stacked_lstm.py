
import tensorflow as tf


# class StackedLSTM(tf.keras.Model):
#   def __init__(self, vocab_size, embedding_dim, n_hidden):
#     super().__init__(self)
#     self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#
#     rnn_cells = [tf.keras.layers.LSTMCell(n_hidden) for _ in range(2)]
#     stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
#     self.lstm_layer = tf.keras.layers.RNN(stacked_lstm)
#
#     self.dense = tf.keras.layers.Dense(vocab_size)
#
#   def call(self, inputs, states=None, return_state=False, training=False):
#     x = inputs
#     x = self.embedding(x, training=training)
#     if states is None:
#       states = self.lstm_layer.get_initial_state(x)
#     x, states = self.lstm_layer(x, initial_state=states, training=training)
#     x = self.dense(x, training=training)
#
#     if return_state:
#       return x, states
#     else:
#       return x
#
#
#
# def build_stacked_lstm():
#     model = StackedLSTM(80, 8, 256)
#     model.call(tf.keras.layers.Input(shape=(80), name="test_prefix"))
#     # model.build(input_shape=(None, 80))
#     model.summary()
#     return model

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras import Sequential


def build_stacked_lstm():
    vocab_size, embedding_dim, n_hidden = 80, 8, 256
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))

    # rnn_cells = [tf.keras.layers.LSTMCell(n_hidden) for _ in range(2)]
    # stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
    # lstm_layer = tf.keras.layers.RNN(stacked_lstm)

    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(LSTM(n_hidden, return_sequences=False))
    model.add(Dense(vocab_size, activation='softmax'))

    return model