import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

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
    embedding_dim = 100
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'

    # 5. Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

    # 6. Defining model with glove

    # You need to download glove.6B.100d.txt
    # wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt -O glove.6B.100d.txt
    embedding_index = {}
    with open('glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Glove is used for embedding layer.
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 7. Compile and training
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 10
    model.fit(padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels))
