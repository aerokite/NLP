from w2v import Word2Vec
import matplotlib.pyplot as plt


def read_data(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            data.append(line)
    return data


corpus = read_data("watch_review_clean.txt")
# The general rule of thumb for embedding_size is (vocabulary_size^0.25)
skip_gram = Word2Vec(window_size=2, embedding_size=8, epochs=100, alpha=0.025)
skip_gram.build_vocab(corpus)

hidden_weight, output_weight, losses = skip_gram.run(check_words=["good", "love", "size", "brother"])
plt.plot(losses)
plt.show()
