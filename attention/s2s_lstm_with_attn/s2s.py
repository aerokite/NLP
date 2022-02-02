import torch

import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
# from torch.utils.tensorboard import SummaryWriter
from models import *

# Set seed for deterministic result
SEED = 1876189809
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

spacy_german = spacy.load("de_core_news_lg")
print(spacy_german._path)
spacy_english = spacy.load("en_core_web_lg")

def tokenize_ger(text):
    return [tok.text for tok in spacy_german.tokenizer(text)][::-1]

def tokenize_eng(text):
    return [tok.text for tok in spacy_english.tokenizer(text)]


german_field = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english_field = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german_field, english_field)
)
german_field.build_vocab(train_data, min_freq=2)
english_field.build_vocab(train_data, min_freq=2)

num_epochs = 100
learning_rate = 0.001
batch_size = 100

input_size = len(german_field.vocab)
output_size = len(english_field.vocab)
embedding_dim = 300
hidden_dim = 1024
clip = 1

encoder = Encoder(input_size, embedding_dim, hidden_dim).to(device)
german_vector = torch.FloatTensor(spacy_german.vocab.vectors.data)
encoder.embedding = nn.Embedding.from_pretrained(german_vector, freeze=False)

decoder = Decoder(output_size, embedding_dim, hidden_dim, output_size).to(device)
english_vector = torch.FloatTensor(spacy_english.vocab.vectors.data)
decoder.embedding = nn.Embedding.from_pretrained(english_vector, freeze=False)

model = Seq2Seq(encoder, decoder, device).to(device)

# This is to use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english_field.vocab.stoi["<pad>"]
# This is to use CrossEntropy loss function.
# Ignore padding entry
loss_func = nn.CrossEntropyLoss(ignore_index=pad_idx)

# This is to use TensorBoard for display loss 
# writer = SummaryWriter("data/runs/loss_plot")
writer = None

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    device=device,
)

# load_checkpoint(torch.load("/content/drive/MyDrive/Colab Notebooks/s2s_lstm_attn/data/checkpoint.pth.tar"), model, optimizer)

test_sentance = "Diese alte Dame schaut sich das Spiel an und isst Süßigkeiten."
process = Process(model, spacy_german, german_field, english_field, optimizer, loss_func, test_sentance, writer, clip, device)

process.run(num_epochs, train_iterator, validation_iterator)


score = bleu(test_data[:], model, spacy_german, german_field, english_field, device)
print(f"Bleu score {score*100:.2f}")
