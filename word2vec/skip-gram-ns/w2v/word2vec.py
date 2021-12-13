import numpy as np
import re
import time
import pickle
from nltk.corpus import stopwords
import numpy_ml as npml
import random


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()


# Tokenize each sentence to get words. Considered only [a-zA-Z]
def tokenize(corpus):
    regex = re.compile('[^a-zA-Z]')
    stop_words = set(stopwords.words('english'))
    data = []
    sent_count = 0
    word_count = 0

    for c in corpus:
        words = c.split()
        if len(words) > 30 or len(words) < 10:
            continue
        x = []
        for word_freq in words:
            word_freq = regex.sub('', word_freq.lower())
            if word_freq == "":
                continue
            if word_freq in stop_words:
                continue
            x.append(word_freq)

        if len(x) < 5:
            continue

        sent_count += 1
        word_count += len(x)
        data.append(x)

    word_freq = {}

    for token_s in data:
        for token in token_s:
            word_freq[token] = word_freq.get(token, 0) + 1

    # Ignore words with freq <=2
    word_freq = {word: freq for word, freq in word_freq.items() if freq >= 10}
    word_list = word_freq.keys()
    word_idx = {w: idx for (idx, w) in enumerate(word_list)}
    idx_word = {idx: w for (idx, w) in enumerate(word_list)}

    print("Total sentences:", sent_count)
    print("Total words:", word_count)

    return data, word_idx, idx_word, word_freq


def training_data(sentences, word2idx, window_size, ):
    pairs = []
    for sentence in sentences:
        sentence_len = len(sentence)
        sentence_map = [word2idx.get(w, -1) for w in sentence]
        for index, idx1 in enumerate(sentence_map):
            if idx1 == -1:
                continue

            s = index - window_size
            e = index + window_size + 1

            # One Hot Vector for a word
            center_vector = [idx1]

            # One Hot Vector list for neighboring words

            for i in range(s, e):
                if 0 <= i < sentence_len and i != index:
                    idx2 = sentence_map[i]
                    if idx2 == -1:
                        continue
                    pairs.append((center_vector, [idx2]))

    return pairs


# Save model for further use
def save(bw1, bw2, word2idx):
    save_data = {
        "bw1": bw1,
        "bw2": bw2,
        "word2idx": word2idx,
    }

    with open("model.save", 'wb') as f:
        pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def get_negative_samples(unigram, center_word, context_words, neg_size):
    negative_context = []
    negative_samples = np.random.choice(len(unigram), size=(neg_size + len(context_words) + 1), p=unigram)
    for ns in negative_samples:

        if len(negative_context) == neg_size:
            break
        if ns in center_word + context_words:
            continue
        negative_context.append(ns)

    return negative_context


class Word2Vec:
    def __init__(self, window_size=2, embedding_size=10, epochs=10, alpha=0.001, neg_size=0):
        self.word_freq = None
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.alpha = alpha
        self.vocabulary_size = 0
        self.neg_size = neg_size

        self.sentences = None
        self.idx2word = None
        self.word2idx = None
        self.opt = npml.neural_nets.optimizers.Adam(lr=alpha)

        np.random.seed(123456)

    # check context words
    def check(self, matrix, word):
        idx1 = self.word2idx.get(word, -1)
        if idx1 == -1:
            print(word + " not found")
            return
        w1 = matrix[idx1, :]

        scores = []
        for word2 in self.word2idx:
            idx2 = self.word2idx.get(word2, 0)
            w2 = matrix[idx2, :]

            norm1 = np.linalg.norm(w1)
            norm2 = np.linalg.norm(w2)
            score = np.dot(w1, w2) / (norm1 * norm2)

            scores.append((word2, '%.5f' % score))

        scores.sort(key=lambda x: x[1], reverse=True)
        print(scores[:5])

    def gen_unigram(self):
        words = list(self.word_freq.keys())
        random.shuffle(words)

        unigram = np.zeros(self.vocabulary_size)
        for word in words:
            freq = self.word_freq[word]
            f = freq ** 0.75
            unigram[self.word2idx[word]] = f
        unigram = unigram / np.sum(unigram)

        return unigram

    def skip_gram_ns(self, train_x, train_y, outsiders, w1, w2):
        c_idx = train_y + outsiders

        h = w1[train_x]
        u = np.dot(w2[c_idx], h.T)

        ep = sigmoid(u)
        # minus 1 for positive word
        ep[0] -= 1

        delta_h = np.dot(ep.T, w2[c_idx])
        delta_o = np.dot(ep, h)

        w1[train_x] = w1[train_x] - self.alpha * delta_h

        for i, d in enumerate(c_idx):
            w2[d] = w2[d] - self.alpha * delta_o[i]

        return w1, w2

    def build_vocab(self, corpus):
        # tokenized sentence, mappings
        sentences, word2idx, idx2word, word_freq = tokenize(corpus)
        self.sentences = sentences
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freq = word_freq

        self.vocabulary_size = len(word2idx)
        print("Total Vocab: ", self.vocabulary_size)

    def run(self, check_words=None):

        # weight matrix
        W1 = np.random.uniform(-0.9, 0.9, (self.vocabulary_size, self.embedding_size)) / self.embedding_size
        W2 = np.random.uniform(-0.9, 0.9, (self.vocabulary_size, self.embedding_size))

        start = time.time()

        # training_data returns tuples of 2 'one hot vector'.
        pairs = training_data(sentences=self.sentences, word2idx=self.word2idx, window_size=self.window_size)
        print("Total pairs: ", len(pairs))
        total_pair = len(pairs)

        for e in range(self.epochs):
            unigram = self.gen_unigram()
            print_progress_bar(0, total_pair, prefix='Progress:', suffix='Complete', length=50)
            p_count = 0
            for center, neighbors in pairs:
                outsiders = get_negative_samples(unigram, center, neighbors, self.neg_size)
                W1, W2 = self.skip_gram_ns(center, neighbors, outsiders, W1, W2)
                if p_count % 1000 == 0:
                    print_progress_bar(p_count, total_pair, prefix='Progress:', suffix='Complete', length=50)
                p_count += 1

            print('\t Elapse time: ', int(time.time() - start), '(s)')
            for cw in check_words:
                self.check(W1, cw)

            self.alpha *= 1.0 / (1.0 + self.alpha * e)
            save(W1, W2, self.word2idx)

        return W1, W2
