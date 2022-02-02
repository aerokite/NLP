import torch
import torch.nn as nn
import random

# Encoder is responsible to represent
# a source sentence into a context state
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, n_layers, dropout_p):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # To Prevent Neural Networks from Overfitting; randomly delete some nueron
        self.dropout = nn.Dropout(dropout_p)
        # This will convert word indices to word-vector
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # This is the RNN model for seq to seq
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_p)

    def forward(self, x):

        # x: (seq_length, batch_size)
        embedded = self.dropout(self.embedding(x))
        # embedded: (seq_length, batch_size, embedding_dim)

        _, (hidden, cell) = self.rnn(embedded)
        # hidden: (n_layers, batch_size, hidden_dim)
        # cell: (n_layers, batch_size, hidden_dim)

        # This states will be passed to decoder
        return hidden, cell


# Decoder is reponsible to generate
# a target sentence from a context state
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size, n_layers, dropout_p):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size


        # To Prevent Neural Networks from Overfitting; randomly delete some nueron
        self.dropout = nn.Dropout(dropout_p)
        # This will convert word indices to word-vector
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # This is the RNN model for seq to seq
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_p)
        # This fully connected layer is to convert hidden layer to output layer
        self.fc = nn.Linear(hidden_dim, output_size)

    # This process a single word at a time for all batch
    def forward(self, x, hidden, cell):

        # This convertes (x: [batch_size]) to (x: [1, batch_size])
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        # embedded: (1, batch_size, embedding_dim)

        # hidden: (n_layers, batch_size, hidden_dim)
        # cell: (n_layers, batch_size, hidden_dim)
        # This RNN also takes hidden & cell state from previous time
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # outputs : (1, batch_size, hidden_dim)

        # This will predict the output for all batch
        predictions = self.fc(outputs)
        # predictions: (1, batch_size, output_size)
        predictions = predictions.squeeze(0)
        # predictions: (batch_size, output_size)

        return predictions, hidden, cell


# Seq2Seq combines Encoder & Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teaching_factor=0.5):

        # source: (source_seq_length, batch_size)
        # target: (target_seq_length, batch_size)

        # Encoder processes the input batch and 
        # returns hidden & cell states
        hidden, cell = self.encoder(source)

        batch_size = source.shape[1]
        output_size = self.decoder.output_size
        target_seq_length = target.shape[0]

        # This is to store predicted output from decoder
        outputs = torch.zeros(target_seq_length, batch_size, output_size).to(self.device)

        # Decoder receives one word at a step.
        x = target[0]

        for t in range(1, target_seq_length):
            # Use hidden & cell state from previous step
            # x: (batch_size)
            # hidden: ()
            # cell: ()
            predictions, hidden, cell = self.decoder(x, hidden, cell)
            # predictions: (batch_size, output_size)

            # Store the predictions
            outputs[t] = predictions

            # Find the best prediction for next step
            best_guess = predictions.argmax(1)

            # It takes from predicted word & target word for next step dandomly
            x = target[t] if random.random() < teaching_factor else best_guess

        return outputs
