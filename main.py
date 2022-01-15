import csv

import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import layers

DATASET = "../reduced-data-set.csv"
SEQUENCE_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 20

articles = ""
# Open and read the csv file containing the data set
# LINK https://docs.python.org/3/library/csv.html
with open(DATASET, newline="", encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    # row: title, content, category
    for row in reader:
        if row[0] == "title":
            continue
        articles += row[1]

# 10000 to make sure all the characters are added. Processing time is fast enough, so a high number doesn't really matter.
vocab = sorted(set(articles))
print("Vocab size:", len(vocab))
string_to_ints = layers.StringLookup(vocabulary=vocab)
# unicode split instead of encode, since we want the unicode code points as an array of strings, and not a single string
raw_data = string_to_ints(tf.strings.unicode_split(articles, input_encoding="UTF-8"))

def ints_to_string(ints):
    # We create a reverse layer by using invert=True, and just reusing the same vocabulary that was set previously.
    layer = layers.StringLookup(vocabulary=vocab, invert=True)
    return tf.strings.reduce_join(layer(ints)).numpy().decode("utf-8")

dataset = tf.data.Dataset.from_tensor_slices(raw_data)
# for single_int in dataset.take(20):
#     print(ints_to_string(single_int))
sequences = dataset.batch(SEQUENCE_LENGTH+1, drop_remainder=True)

# for seq in sequences.take(3):
#     print(ints_to_string(seq))

def map_func(seq):
    # All but the last, all but the first. This serves as input and target for predictions.
    return seq[:-1], seq[-1]

dataset = sequences.map(map_func)
# for input, target in dataset.take(3):
#     print("Input :", ints_to_string(input))
#     print("Target:", ints_to_string(target))
dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True).prefetch(-1)

vocab_length = len(vocab)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()

print("Initial preprocessing done, building model...")

# Create the model
model = keras.Sequential()
model.add(layers.Embedding(vocab_length, 256))
model.add(layers.LSTM(1028))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(SEQUENCE_LENGTH))
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

# class TextModel(keras.Model):
#     def __init__(self, vocab_length, output_dimension, units):
#         super().__init__(self)
#         self.embedding = layers.Embedding(vocab_length, output_dimension)
#         self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
#         self.dense = layers.Dense(vocab_length)

#     def call(self, input, state=None):
#         output = self.embedding(input)
#         if state == None:
#             state = self.lstm.get_initial_state(input)
#         output, state = self.lstm(output, initial_state=state)
#         output = self.dense(input)

#         return output, state

# model = TextModel(vocab_length, 256, 512)
# model.compile(loss="sparse_categorical_crossentropy")
# model.build((None, 256))

# model.summary()


# class MyModel(tf.keras.Model):

#     def __init__(self, vocab_size, embedding_dim, rnn_units):
#         super().__init__(self)
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#         self.gru = tf.keras.layers.GRU(rnn_units,
#                                        return_sequences=True,
#                                        return_state=True)
#         self.dense = tf.keras.layers.Dense(vocab_size)

#     def call(self, inputs, states=None, return_state=False, training=False):
#         x = inputs
#         x = self.embedding(x, training=training)
#         if states is None:
#             states = self.gru.get_initial_state(x)
#         x, states = self.gru(x, initial_state=states, training=training)
#         x = self.dense(x, training=training)

#         if return_state:
#             return x, states
#         else:
#             return x


# model = MyModel(vocab_size=vocab_length, embedding_dim=256, rnn_units=512)
# model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

print(f"Model made, fitting model with {EPOCHS} epochs...")

model.fit(dataset, epochs=EPOCHS)

print("Model fitted, generating some text...")

# for input_example_batch, target_example_batch in dataset.take(1):
input_string = "hallo, mijn naam is Karin en ik ben naar Nederland gefietst omdat ik het hier mooi vindt. Vervolgens"
result = ""
for _ in range(20):
    tensor = string_to_ints(tf.strings.unicode_split(input_string[len(result):] + result, input_encoding="UTF-8"))
    tensor = tf.expand_dims(tensor, 0)
    prediction = model(tensor)
    prediction = tf.squeeze(prediction)
    print(prediction)
    index = tf.math.argmax(prediction)
    print(index)
    result += ints_to_string(index)

print(input_string + result)

# example_batch_loss = loss(target_example_batch, example_batch_predictions)
# mean_loss = example_batch_loss.numpy().mean()
# print("Prediction shape: ", example_batch_predictions.shape,
#       " # (batch_size, sequence_length, vocab_length)")
# print("Mean loss:        ", mean_loss)
# print(tf.exp(mean_loss).numpy())
