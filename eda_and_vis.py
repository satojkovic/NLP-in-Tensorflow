import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


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
    positive_sentences = [sentence for sentence, label in zip(all_sentences, all_labels) if label == 1]
    negative_sentences = [sentence for sentence, label in zip(all_sentences, all_labels) if label == 0]
    positive_sentence_length = [len(sentence.split(' ')) for sentence in positive_sentences]
    negative_sentence_length = [len(sentence.split(' ')) for sentence in negative_sentences]
    max_length = max(max(positive_sentence_length), max(negative_sentence_length))

    # 5. Plot the distribution
    plt.hist(positive_sentence_length, bins=max_length, alpha=0.5, rwidth=2, label='positive')
    plt.hist(negative_sentence_length, bins=max_length, alpha=0.5, rwidth=2, label='negative')
    plt.xlabel('Review length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # 6. Count the number of words
    def get_top_n_words(corpus, n=20, is_stop_words=False):
        vectorizer = CountVectorizer(stop_words='english') if is_stop_words else CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        X_sum = X.sum(axis=0)
        words_freq = [(word, X_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    top_n = 20
    positive_words_freq = get_top_n_words(positive_sentences, top_n)
    negative_words_freq = get_top_n_words(negative_sentences, top_n)
    positive_words_freq_wo_stop_words = get_top_n_words(positive_sentences, top_n, is_stop_words=True)
    negative_words_freq_wo_stop_words = get_top_n_words(negative_sentences, top_n, is_stop_words=True)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    left = range(top_n)
    plt.subplot(2, 2, 1)
    plt.title('Positive')
    plt.bar(left, [freq for _, freq in positive_words_freq], tick_label=[word for word, _ in positive_words_freq], color='r', width=0.6)
    plt.xticks(rotation=-90, fontsize=8)

    plt.subplot(2, 2, 2)
    plt.title('Negative')
    plt.bar(left, [freq for _, freq in negative_words_freq], tick_label=[word for word, _ in negative_words_freq], color='b', width=0.6)
    plt.xticks(rotation=-90, fontsize=8)

    plt.subplot(2, 2, 3)
    plt.title('Positive without stop words')
    plt.bar(left, [freq for _, freq in positive_words_freq_wo_stop_words], 
        tick_label=[word for word, _ in positive_words_freq_wo_stop_words], color='r', width=0.6)
    plt.xticks(rotation=-90, fontsize=8)

    plt.subplot(2, 2, 4)
    plt.title('Negative without stop words')
    plt.bar(left, [freq for _, freq in negative_words_freq_wo_stop_words], 
        tick_label=[word for word, _ in negative_words_freq_wo_stop_words], color='b', width=0.6)
    plt.xticks(rotation=-90, fontsize=8)

    plt.show()
