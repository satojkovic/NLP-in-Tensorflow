import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Load imdb dataset
    imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

    # 2. Split imdb dataset into train / test data
    train_data, test_data = imdb['train'], imdb['test']

    # 3. Prepare sentences and labels
    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(str(s.numpy()))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(str(s.numpy()))
        testing_labels.append(l.numpy())

    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    # 4. Hyperparameters
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_token = '<OOV>'

    # 5. Tokenize
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

    # 6. Defining model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 7. Compile and training
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 10
    history = model.fit(padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels))

    # 8. Plot the history of training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.show()