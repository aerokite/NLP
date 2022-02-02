from unicodedata import bidirectional
import torch
import torch.nn as nn
import random

# Encoder is responsible to represent
# a source sentence into a context state
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        # This will convert word indices to word-vector
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # This is the RNN model for seq to seq
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, x):

        # x: (seq_length, batch_size)
        embedded = self.embedding(x)
        # embedded: (seq_length, batch_size, embedding_dim)

        states, (hidden, cell) = self.rnn(embedded)
        # states: (seq_length, batch_size, hidden_dim*2)
        # hidden: (2, batch_size, hidden_dim)
        # cell: (2, batch_size, hidden_dim)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        # hidden: (1, batch_size, hidden_dim)
        cell = self.fc_cell(torch.cat((cell[:1], cell[1:2]), dim=2))
        # cell: (1, batch_size, hidden_dim)

        # This states will be passed to decoder
        return states, hidden, cell


# Decoder is reponsible to generate
# a target sentence from a context state
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # This will convert word indices to word-vector
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # hidden_dim*2(encoding_state) + hidden_dim(decoder_hidden)
        self.energy = nn.Linear(hidden_dim*3, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

        # This is the RNN model for seq to seq
        self.rnn = nn.LSTM(hidden_dim*2 + embedding_dim, hidden_dim)
        # This fully connected layer is to convert hidden layer to output layer
        self.fc = nn.Linear(hidden_dim, output_size)

    # This process a single word at a time for all batch
    def forward(self, x, encoder_states, hidden, cell):

        # This convertes (x: [batch_size]) to (x: [1, batch_size])
        x = x.unsqueeze(0)
        embedded = self.embedding(x)
        # embedded: (1, batch_size, embedding_dim)

        # encoder_states: (seq_length, batch_size, hidden_dim*2)
        # hidden: (1, batch_size, hidden_dim)
        # cell: (1, batch_size, hidden_dim)

        encoder_seq_length = encoder_states.shape[0]
        h_reshape = hidden.repeat(encoder_seq_length, 1, 1)
        # h_reshape: (seq_length, batch_size, hidden_dim)

        # (seq_length, batch_size, hidden_dim*3) -> (seq_length, batch_size, 1)
        energy = self.energy(torch.cat((h_reshape, encoder_states), dim=2))
        # energy: (seq_length, batch_size, 1)
        attention = self.softmax(self.relu(energy))
        # attention: (seq_length, batch_size, 1)
        attention = attention.permute(1, 2, 0)
        # attention: (batch_size, 1, seq_length)

        # encoder_states: (seq_length, batch_size, hidden_dim*2)
        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states: (batch_size, seq_length, hidden_dim*2)

        # (b×n×m)*(b×m×p)=(b×n×p).permute(1,0,2)=(n,b,p)
        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2)
        # context_vector: (1, batch_size, hidden_dim*2)
        # embedded: (1, batch_size, embedding_dim)
        input = torch.cat((context_vector, embedded), dim=2)
        # input: (1, batch_size, hidden_dim*2+embedding_dim)

        # hidden: (1, batch_size, hidden_dim)
        # cell: (1, batch_size, hidden_dim)
        # This RNN also takes hidden & cell state from previous time
        outputs, (hidden, cell) = self.rnn(input, (hidden, cell))
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
        # returns encoder_states, hidden & cell states
        encoder_states, hidden, cell = self.encoder(source)
        # encoder_states: (seq_length, batch_size, hidden_dim*2)
        # hidden: (1, batch_size, hidden_dim)
        # cell: (1, batch_size, hidden_dim)

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
            predictions, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            # predictions: (batch_size, output_size)
            # hidden: (1, batch_size, hidden_dim)
            # cell: (1, batch_size, hidden_dim)

            # Store the predictions
            outputs[t] = predictions

            # Find the best prediction for next step
            best_guess = predictions.argmax(1)

            # It takes from predicted word & target word for next step dandomly
            x = target[t] if random.random() < teaching_factor else best_guess

        return outputs
