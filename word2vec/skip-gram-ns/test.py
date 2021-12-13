import numpy as np
import sys
import pickle
from nltk.corpus import stopwords

with open("trained-model", 'rb') as fp:
    load_data = pickle.load(fp)
    BW1 = load_data["bw1"]
    BW2 = load_data["bw2"]
    W1 = BW1
    W2 = BW2
    word2idx = load_data["word2idx"]


def similarity(word, n=5):
    m = {}

    idx1 = word2idx.get(word, -1)
    if idx1 == -1:
        print(word + " missing")
        return -1

    for w in word2idx:
        if w == word:
            continue

        idx2 = word2idx.get(w)
        w1 = W1[idx1, :]
        w2 = W1[idx2, :]

        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        score = np.dot(w1, w2) / (norm1 * norm2)

        m[w] = score

    m = sorted(m.items(), key=lambda item: item[1], reverse=True)

    c = 0
    print(n)
    for w in m:
        c += 1
        print(w)
        if c > n:
            break


def similarity2(word1, word2, word3, n=5):
    m = {}

    idx1 = word2idx.get(word1, -1)
    if idx1 == -1:
        print(word1 + " missing")
        return -1

    idx2 = word2idx.get(word2, -1)
    if idx2 == -1:
        print(word2 + " missing")
        return -1

    idx3 = word2idx.get(word3, -1)
    if idx3 == -1:
        print(word3 + " missing")
        return -1

    for tw in word2idx:
        if tw == word1 or tw == word2 or tw == word3:
            continue

        idx4 = word2idx.get(tw)

        w1 = W1[idx1, :]
        w2 = W1[idx2, :]
        w3 = W1[idx3, :]
        w4 = W1[idx4, :]

        w = w1-w2+w3

        norm = np.linalg.norm(w)
        norm4 = np.linalg.norm(w4)
        score = np.dot(w, w4) / (norm * norm4)

        m[tw] = score

    m = sorted(m.items(), key=lambda item: item[1], reverse=True)

    c = 0
    for w in m:
        c += 1
        print(w)
        if c > n:
            break


if len(sys.argv) == 2:
    similarity(sys.argv[1])

if len(sys.argv) == 4:
    similarity2(sys.argv[1], sys.argv[2], sys.argv[3])

