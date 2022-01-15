import csv
from datetime import datetime

import keras
import tensorflow as tf
from keras.callbacks import TensorBoard
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
sequences = dataset.batch(SEQUENCE_LENGTH+1, drop_remainder=True)

def map_func(seq):
    # All but the last, all but the first. This serves as input and target for predictions.
    return seq[:-1], seq[-1]

dataset = sequences.map(map_func)
dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True).prefetch(-1)

vocab_length = len(vocab)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()

print("Initial preprocessing done, building model...")

# Create the model
model = keras.Sequential()
model.add(layers.Embedding(vocab_length, 256))
model.add(layers.LSTM(1028, return_sequences=True))
model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(1028))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(SEQUENCE_LENGTH))
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

print(f"Model made, fitting model with {EPOCHS} epochs...")

log_dir = "logs\\" + datetime.now().strftime("%d-%m-%YT%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
start = datetime.now()

model.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard])

print(f"Model fitted, took {datetime.now() - start}")

print("Model fitted, generating some text...")

# for input_example_batch, target_example_batch in dataset.take(1):
input_string = "hallo, mijn naam is Karin en ik ben naar Nederland gefietst omdat ik het hier mooi vind. Vervolgens"
result = ""
for _ in range(400):
    tensor = string_to_ints(tf.strings.unicode_split(input_string[len(result):] + result, input_encoding="UTF-8"))
    tensor = tf.expand_dims(tensor, 0)
    prediction = model(tensor)
    prediction = tf.squeeze(prediction)
    index = tf.math.argmax(prediction)
    print(index)
    result += ints_to_string(index)

print(input_string + result)
