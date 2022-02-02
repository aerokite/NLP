### I have used spacy dataset for low volume

    Sentences:
       Training:   29000
       Validating: 1014
       Testing:    1000
    Vocabulary:
       German:     7853
       English:    5893

### Settings

    learning_rate = 0.0005
    batch_size = 100
    d_model = 300
    n_layers = 2
    n_heads = 4
    ffn_dim = 1200

### Result

    test_sentence = "Die alte Frau sieht sich das Fußballspiel an und isst Süßigkeiten."
    test_translation = "The old woman is watching the football match and eating candy."

    Epoch: 01 | Time: 0m 15s
        Train Loss: 3.811 | Train Exp:  45.217
        Val. Loss: 2.776 |  Val. Exp:  16.047
        Learning Rate: 0.0005000
    Translated example sentence: 
    the woman is about to be the ocean and <unk> <unk> .

    Epoch: 02 | Time: 0m 15s
        Train Loss: 2.505 | Train Exp:  12.244
        Val. Loss: 2.222 |  Val. Exp:   9.226
        Learning Rate: 0.0004250
    Translated example sentence: 
    the old woman is looking at the soccer game .

    Epoch: 03 | Time: 0m 15s
        Train Loss: 2.022 | Train Exp:   7.553
        Val. Loss: 1.972 |  Val. Exp:   7.188
        Learning Rate: 0.0003612
    Translated example sentence: 
    the old woman is eating the soccer game .

    Epoch: 04 | Time: 0m 15s
        Train Loss: 1.725 | Train Exp:   5.615
        Val. Loss: 1.835 |  Val. Exp:   6.268
        Learning Rate: 0.0003071
    Translated example sentence: 
    the old woman is looking at the soccer game .

    Epoch: 05 | Time: 0m 15s
        Train Loss: 1.505 | Train Exp:   4.506
        Val. Loss: 1.762 |  Val. Exp:   5.822
        Learning Rate: 0.0002610
    Translated example sentence: 
    the old woman is looking at the soccer game .

    Epoch: 06 | Time: 0m 15s
        Train Loss: 1.337 | Train Exp:   3.807
        Val. Loss: 1.718 |  Val. Exp:   5.573
        Learning Rate: 0.0002219
    Translated example sentence: 
    the old woman is looking at the soccer game .

    Epoch: 07 | Time: 0m 15s
        Train Loss: 1.201 | Train Exp:   3.323
        Val. Loss: 1.696 |  Val. Exp:   5.453
        Learning Rate: 0.0001886
    Translated example sentence: 
    the old woman is looking at the soccer ball and eating .

    Epoch: 08 | Time: 0m 15s
        Train Loss: 1.092 | Train Exp:   2.981
        Val. Loss: 1.687 |  Val. Exp:   5.406
        Learning Rate: 0.0001603
    Translated example sentence: 
    the old woman is eating the soccer game .

    Epoch: 09 | Time: 0m 15s
        Train Loss: 1.000 | Train Exp:   2.718
        Val. Loss: 1.688 |  Val. Exp:   5.409
        Learning Rate: 0.0001362
    Translated example sentence: 
    the old woman is looking at the soccer game .

    Epoch: 10 | Time: 0m 15s
        Train Loss: 0.924 | Train Exp:   2.521
        Val. Loss: 1.691 |  Val. Exp:   5.427
        Learning Rate: 0.0001158
    Translated example sentence: 
    the old woman is looking at the soccer ball and eating .

    Epoch: 11 | Time: 0m 15s
        Train Loss: 0.862 | Train Exp:   2.369
        Val. Loss: 1.699 |  Val. Exp:   5.468
        Learning Rate: 0.0000984
    Translated example sentence: 
    the old woman is looking at the soccer game and eating .

    Bleu score 37.47
