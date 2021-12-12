from w2v import Word2Vec
import matplotlib.pyplot as plt


def read_data(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            data.append(line)
    return data


corpus = read_data("phone_review_clear.txt")
# The general rule of thumb for embedding_size is (vocabulary_size^0.25)
skip_gram = Word2Vec(window_size=2, embedding_size=64, epochs=100, alpha=0.001, neg_size=10)
skip_gram.build_vocab(corpus)

hidden_weight, output_weight = skip_gram.run(check_words=["good", "red", "samsung", "brother"])
