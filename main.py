import csv
import os
import sys
from datetime import datetime

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import layers

DATASET = "../reduced-data-set.csv"
SEQUENCE_LENGTH = 100
BATCH_SIZE = 256
EPOCHS = 100

articles = ""
# Open and read the csv file containing the data set
with open(DATASET, newline="", encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    # row: title, content, category
    for row in reader:
        if row[0] == "title":
            continue
        articles += row[1]

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
    # All but the last, the last. This serves as input and target for predictions.
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
model.add(layers.LSTM(768, return_sequences=True))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(768, return_sequences=True))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(768))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(SEQUENCE_LENGTH))
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.summary()

if len(sys.argv) > 1:
    print(f"\n\nCommand line argument found, importing model from path \"{sys.argv[1]}\"\n\n")
    model.load_weights(sys.argv[1])
    print("Model restored. Evaluating...")
    # model.evaluate(dataset)
    print("Model evaluated. Generating text...\n\n")
else:
    print(f"\n\nModel made, fitting model with {EPOCHS} epochs...\n\n")

    datetime_string = datetime.now().strftime("%d-%m-%YT%H-%M-%S")
    log_dir = "logs\\" + datetime_string
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = os.path.join("./checkpoints", "ckpt_e{epoch}_l{loss}")
    checkpoints = ModelCheckpoint(checkpoint_path, save_weights_only=True)

    start = datetime.now()

    model.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard, checkpoints])

    print(f"\n\nModel fitted, took {datetime.now() - start}. Saving model...")

    model.save(f"saved_models\\model_{datetime_string}")

    print("Model saved, generating some text...\n\n")

start = datetime.now()

input_string = "Hoewel het nog niet duidelijk is wie de moord gepleegd heeft, is de politie"
result = ""
for i in range(800):
    tensor = string_to_ints(tf.strings.unicode_split(input_string[len(result):] + result, input_encoding="UTF-8"))
    tensor = tf.expand_dims(tensor, 0)
    prediction = model(tensor)
    prediction = tf.squeeze(prediction)
    index = tf.math.argmax(prediction)
    
    ### Single word prediction
    # if ints_to_string(index) == " ":
    #     break

    result += ints_to_string(index)

    if i % 200 == 0 and i > 0:
        print(input_string + result)

print(f"\nAll text generated, took {datetime.now() - start}\n\nResult: " + input_string + result)
