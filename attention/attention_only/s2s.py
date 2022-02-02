import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
import numpy as np
import spacy
import random
from models import Encoder, Decoder, Seq2Seq, Process
from utils import bleu


SEED = 1876189809
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# I have used spacy dataset for low volume
# Sentences:
#   Training:   29000
#   Validating: 1014
#   Testing:    1000
# Vocabulary:
#   German:     7853
#   English:    5893
spacy_german = spacy.load("de_core_news_lg")
spacy_english = spacy.load("en_core_web_lg")


def tokenize_ger(text):
    return [tok.text for tok in spacy_german.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_english.tokenizer(text)]


# This is used to tokenize and append extra token
german_field = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>", batch_first=True)
english_field = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>", batch_first=True)


# To load data from local file
# fields = {"src": ("src", german_field), "trg": ("trg", english_field)}
# train_data, valid_data, test_data = TabularDataset.splits(
#     path="/content", train="train.txt", validation="val.txt", test="test.txt", format="json", fields=fields
# )


train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german_field, english_field)
)


# This will generate vocabulary with minimum freq.
german_field.build_vocab(train_data, min_freq=2)
english_field.build_vocab(train_data, min_freq=2)
source_vocab_size = len(german_field.vocab)
target_vocab_size = len(english_field.vocab)

num_epochs = 100
learning_rate = 0.0005
# Large batch size exceeds free GPU memory
batch_size = 100
# Dimension for all matrix
# I will call it hidden_dim & embedding_dim as well
d_model = 300
# Number of repeated layers
n_layers = 2
# Number of heads
n_heads = 4
ffn_dim = 1200

# Use this to clip gradient norm to avoid exploding
clip = 1
dropout = 0.1

source_pad_idx = german_field.vocab.stoi[german_field.pad_token]
target_pad_idx = english_field.vocab.stoi[english_field.pad_token]

encoder = Encoder(source_vocab_size, d_model, n_layers, n_heads, ffn_dim, source_pad_idx, dropout=dropout).to(device)
decoder = Decoder(target_vocab_size, d_model, n_layers, n_heads, ffn_dim, source_pad_idx, target_pad_idx, device, dropout=dropout).to(device)
# german_vector = torch.FloatTensor(spacy_german.vocab.vectors.data)
# encoder.embedding = nn.Embedding.from_pretrained(german_vector, freeze=False)
# english_vector = torch.FloatTensor(spacy_english.vocab.vectors.data)
# decoder.embedding = nn.Embedding.from_pretrained(english_vector, freeze=False)


model = Seq2Seq(encoder, decoder).to(device)

# This is to use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.95)


# This is to use CrossEntropy loss function.
# Ignore padding entry
loss_func = nn.CrossEntropyLoss(ignore_index=target_pad_idx)

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_key = lambda x: len(x.src),
    sort_within_batch=True,
    device=device,
)


test_sentence = "Die alte Frau sieht sich das Fußballspiel an und isst Süßigkeiten."
test_translation = "The old woman is watching the football game and eating sweets."

process = Process(model, spacy_german, german_field, english_field, optimizer, scheduler, loss_func, test_sentence, clip, device)

process.run(num_epochs, train_iterator, validation_iterator)

score = bleu(test_data, model, spacy_german, german_field, english_field, device)
print(f"Bleu score {score*100:.2f}")