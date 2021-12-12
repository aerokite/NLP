## Word2Vec with SkipGram

Used negative sampling.

### Data

    dataset: Phone Review from Amazon
    filter: word count from 10 to 30, removed stop word & other than [a-z]
    sentences: 63276
    words: 764420
    vocab: 3563

### Setup

    window_size=2
    embedding_size=64
    epochs=4
    alpha=0.001
    negative_sample=10
    frequency_filter=<10

### Output

    good:       cheap 0.95592       excellent 0.95349   solid 0.94612   great 0.94552
    red:        green 0.97566       purple 0.97022      black 0.96274   pink 0.95776
    samsung:    galaxy 0.97316      note 0.96677        nexus 0.95504   tab 0.92893
    brother:    girlfriend 0.97803  boyfriend 0.97680   mother 0.97613  sister 0.97491
