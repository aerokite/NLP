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
        Train Loss: 3.831 | Train Exp:  46.098
        Val. Loss: 2.854 |  Val. Exp:  17.349
        Learning Rate: 0.0005000
    Translated example sentence: 
    the old woman is looking at the park and holding her face .

    Epoch: 02 | Time: 0m 15s
        Train Loss: 2.561 | Train Exp:  12.947
        Val. Loss: 2.257 |  Val. Exp:   9.553
        Learning Rate: 0.0004750
    Translated example sentence: 
    the old woman is looking at the soccer game .

    Epoch: 04 | Time: 0m 15s
        Train Loss: 1.786 | Train Exp:   5.965
        Val. Loss: 1.860 |  Val. Exp:   6.425
        Learning Rate: 0.0004287
    Translated example sentence: 
    the old woman is looking at the soccer game .

    Epoch: 05 | Time: 0m 15s
        Train Loss: 1.562 | Train Exp:   4.768
        Val. Loss: 1.779 |  Val. Exp:   5.925
        Learning Rate: 0.0004073
    Translated example sentence: 
    the old woman is looking at the soccer ball and eating .

    Epoch: 07 | Time: 0m 15s
        Train Loss: 1.229 | Train Exp:   3.416
        Val. Loss: 1.721 |  Val. Exp:   5.592
        Learning Rate: 0.0003675
    Translated example sentence: 
    the old woman is looking at the soccer ball and eating .

    Epoch: 08 | Time: 0m 15s
        Train Loss: 1.095 | Train Exp:   2.989
        Val. Loss: 1.725 |  Val. Exp:   5.612
        Learning Rate: 0.0003492
    Translated example sentence: 
    the old woman is at the soccer game , as she is eating .

    Epoch: 10 | Time: 0m 15s
        Train Loss: 0.873 | Train Exp:   2.393
        Val. Loss: 1.741 |  Val. Exp:   5.701
        Learning Rate: 0.0003151
    Translated example sentence: 
    the old woman is looking at the soccer ball and eating candy .

    Epoch: 12 | Time: 0m 15s
        Train Loss: 0.697 | Train Exp:   2.008
        Val. Loss: 1.817 |  Val. Exp:   6.154
        Learning Rate: 0.0002844
    Translated example sentence: 
    the old woman is watching the soccer game while eating candy .

    Bleu score 36.29

