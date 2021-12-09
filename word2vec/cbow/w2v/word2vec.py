import numpy as np
import re
import time
import pickle
from nltk.corpus import stopwords


# Tokenize each sentence to get words. Considered only [a-zA-Z]
def tokenize(corpus):
    regex = re.compile('[^a-zA-Z]')
    stop_words = set(stopwords.words('english'))
    data = []

    for c in corpus:
        words = c.split()
        x = []
        for word in words:
            word = regex.sub('', word.lower())
            if word == "":
                continue
            if word in stop_words:
                continue
            x.append(word)

        data.append(x)

    word = {}

    for token_s in data:
        for token in token_s:
            word[token] = word.get(token, 0) + 1

    # Ignore words with freq <=2
    word = {word: freq for word, freq in word.items() if freq >= 3}
    word_list = word.keys()
    word_idx = {w: idx for (idx, w) in enumerate(word_list)}
    idx_word = {idx: w for (idx, w) in enumerate(word_list)}

    return data, word_idx, idx_word


def training_data(sentences, word2idx, window_size):
    vocab_siz = len(word2idx)
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
            center_vector = [0] * vocab_siz
            center_vector[idx1] = 1

            # One Hot Vector list for neighboring words
            context = []
            for i in range(s, e):
                if 0 <= i < sentence_len and i != index:
                    idx2 = sentence_map[i]
                    if idx2 == -1:
                        continue
                    context_vector = [0] * vocab_siz
                    context_vector[idx2] = 1
                    context.append(context_vector)
            if len(context) == 0:
                continue

            pairs.append((center_vector, context))
    return pairs


# Save model for further use
def save(bw1, bw2, word2idx, loss_list):
    save_data = {
        "bw1": bw1,
        "bw2": bw2,
        "word2idx": word2idx,
        "loss": loss_list,
    }

    with open("model.save", 'wb') as f:
        pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Word2Vec:
    def __init__(self, window_size=2, embedding_size=10, epochs=10, alpha=0.001):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.alpha = alpha
        self.vocabulary_size = 0

        self.sentences = None
        self.idx2word = None
        self.word2idx = None

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

    def cbow(self, train_x, train_y, w1, w2):
        # forward propagation
        x = np.mean(np.array([tx for tx in train_x]), axis=0)
        h = np.dot(w1.T, x)
        u = np.dot(w2.T, h)
        yp = softmax(u)

        # error calculation
        ep = np.subtract(yp, train_y)

        # backward propagation
        delta_h = np.outer(x, np.dot(w2, ep))
        delta_o = np.outer(h, ep)
        nw1 = w1 - self.alpha * delta_h
        nw2 = w2 - self.alpha * delta_o

        # loss calculation
        loss = -u[train_y.index(1)] + np.log(np.sum(np.exp(u)))
        return nw1, nw2, loss

    def build_vocab(self, corpus):
        # tokenized sentence, mappings
        sentences, word2idx, idx2word = tokenize(corpus)
        self.sentences = sentences
        self.word2idx = word2idx
        self.idx2word = idx2word

        self.vocabulary_size = len(word2idx)
        print("Total Vocab: ", self.vocabulary_size)

    def run(self, check_words=None):

        # weight matrix
        W1 = np.random.uniform(-0.9, 0.9, (self.vocabulary_size, self.embedding_size))
        W2 = np.random.uniform(-0.9, 0.9, (self.embedding_size, self.vocabulary_size))

        start = time.time()
        no_improvement = 0
        losses = []
        best_loss = 1e100

        # training_data returns tuples of 2 'one hot vector'.
        pairs = training_data(self.sentences, self.word2idx, self.window_size)
        print("Total pairs: ", len(pairs))

        for e in range(self.epochs):
            e_loss = 0.

            for center, neighbors in pairs:
                W1, W2, loss = self.cbow(neighbors, center, W1, W2)
                e_loss += loss

            print('\t Epoch: %d' % (e + 1), 'Loss: %f' % e_loss, 'Elapse time: ', int(time.time() - start), '(s)')

            for cw in check_words:
                self.check(W2.T, cw)

            self.alpha *= 1.0 / (1.0 + self.alpha * e)

            if e_loss < 0:
                break
            if e_loss < best_loss:
                losses.append(e_loss)
                best_loss = e_loss
                save(W1, W2, losses, self.word2idx)
                no_improvement = 0
            else:
                no_improvement += 1
                print('\t no improvement')

        return W1, W2, losses
