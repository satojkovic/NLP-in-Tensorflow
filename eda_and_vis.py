import tensorflow_datasets as tfds
import numpy as np
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

    # 4. Count the length of sentences
    all_sentences = training_sentences + testing_sentences
    all_labels = training_labels + testing_labels
    positive_sentence_length = [len(sentence.split(' ')) for sentence, label in zip(all_sentences, all_labels) if label == 1]
    negative_sentence_length = [len(sentence.split(' ')) for sentence, label in zip(all_sentences, all_labels) if label == 0]
    max_length = max(max(positive_sentence_length), max(negative_sentence_length))

    # 5. Plot the distribution
    plt.hist(positive_sentence_length, bins=max_length, alpha=0.5, rwidth=2, label='positive')
    plt.hist(negative_sentence_length, bins=max_length, alpha=0.5, rwidth=2, label='negative')
    plt.xlabel('Review length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

