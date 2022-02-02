from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import get_padding_mask, get_subsequent_mask


class Embedder(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(input_size, d_model)
    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length = 100):
        super().__init__()
        self.d_model = d_model
        positional_data = torch.zeros(max_length, d_model)

        def get_positional_value(pos):
            return [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]

        positional_data = np.array([get_positional_value(pos) for pos in range(max_length)])
        positional_data[:, 0::2] = np.sin(positional_data[:, 0::2])  # dim 2i
        positional_data[:, 1::2] = np.cos(positional_data[:, 1::2])  # dim 2i+1

        positional_data = torch.Tensor(positional_data).unsqueeze(0)
        self.register_buffer('pe', positional_data)
 
    def forward(self, x):
        seq_length = x.size(1)
        x = x + Variable(self.pe[:,:seq_length], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        # Divide vector into equal n_heads part
        d_h = d_model // n_heads
        self.d_h = d_h

        # Following linear models produce Query, Key & Value for words
        # d_model -> d_h * n_heads
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Linear(d_model, d_model)


    def forward(self, Q, K, V, masked=None):
        # Q: (batch_size, seq_length, d_model) -- d_model == embedding_dim
        # K: (batch_size, seq_length, d_model)
        # V: (batch_size, seq_length, d_model)

        batch_size = Q.size(0)
 
        Q = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_h).transpose(1, 2)
        # Q: (batch_size, seq_length, n_heads, d_h) -> (batch_size, n_heads, seq_length, d_h)
        K = self.WK(K).view(batch_size, -1, self.n_heads, self.d_h).transpose(1, 2)
        # K: (batch_size, seq_length, n_heads, d_h) -> (batch_size, n_heads, seq_length, d_h)
        V = self.WV(V).view(batch_size, -1, self.n_heads, self.d_h).transpose(1, 2)
        # V: (batch_size, seq_length, n_heads, d_h) -> (batch_size, n_heads, seq_length, d_h)

        # Q: (batch_size, n_heads, seq_length, d_h)
        # K: (batch_size, n_heads, seq_length, d_h) -> (batch_size, n_heads, d_h, seq_length)
        # Formula: (a,b,c,d)*(a,b,d,f) -> (a,b,c,f)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_h)
        # scores: (batch_size, n_heads, seq_length, seq_length)
        # This is actually calculating word-by-word score. Thats why shape is (-, -, seq_length, seq_length)


        if masked is not None:
            # pad_masked: (batch_size, seq_length, seq_length)
            masked = masked.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            # pad_masked: (batch_size, n_heads, seq_length, seq_length)
            # The shape of the mask is exactly same as scores.
            scores = scores.masked_fill(masked, -1e9)
            # scores: (batch_size, n_heads, seq_length, seq_length)

        attention = self.softmax(scores)

        # attention: (batch_size, n_heads, seq_length, seq_length)
        # V: (batch_size, n_heads, seq_length, d_h)
        context = torch.matmul(attention, V).transpose(1, 2).contiguous()
        # context: (batch_size, n_heads, seq_length, d_h) -> (batch_size, seq_length, n_heads, d_h)

        # The following part concat several heads into one
        output = context.view(batch_size, -1, self.n_heads * self.d_h)
        # output: (batch_size, seq_length, d_model)
        output = self.linear(output)
        # output: (batch_size, seq_length, d_model)

        return output


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, ffn_dim,):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, ffn_dim)
        self.l2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x):
        output = self.l1(x)
        output = torch.relu(output)
        output = self.l2(output)
        return output


# Layer of Encoder
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # The first sub-layer of Encoder. Its a multi-head self-attention.
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.self_attention_norm = nn.LayerNorm(d_model)
        # The second sub-layer of Encoder. Its a positionwise fully connected feed-forward network.
        self.ff_layer = PoswiseFeedForwardNet(d_model, ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, source, pad_masked):
        # As we are feeding same input as query, key & value, its called self-attention.
        # source: (batch_size, seq_length, d_model)
        # pad_masked: (batch_size, seq_length, seq_length)
        output_from_attn = self.self_attention(source, source, source, pad_masked)
        output_from_attn = self.dropout(output_from_attn)
        output_from_attn = self.self_attention_norm(source + output_from_attn)
        # output_from_attn: (batch_size, seq_length, d_model)
        output_from_ffn = self.ff_layer(output_from_attn)
        output_from_ffn = self.dropout(output_from_ffn)
        output_from_ffn = self.ff_layer_norm(output_from_attn + output_from_ffn)
        # output_from_ffn: (batch_size, seq_length, d_model)

        return output_from_ffn


# Encoder is responsible to represent
# a source sentence into a context state
class Encoder(nn.Module):
    def __init__(self, source_vocab_size, d_model, n_layers, n_heads, ffn_dim, source_pad_idx, dropout=0.1) :
        super(Encoder, self).__init__()

        self.source_pad_idx = source_pad_idx

        # It's like Word2Vec
        self.embedding = nn.Embedding(source_vocab_size, d_model)
        # This is used to add position data with embedded words.
        self.position_embedding = PositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)

        # Series of similar layers.
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)])

    def forward(self, source):
        # This method will add masking to padded position (<pad>).
        # source: (batch_size, seq_length)
        pad_masked = get_padding_mask(source, source, self.source_pad_idx)
        # pad_masked: (batch_size, seq_length, seq_length)

        embedded = self.embedding(source)
        # embedded: (batch_size, seq_length, embedding_dim)
        embedded = self.position_embedding(embedded)
        # embedded: (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)

        # The first input to the series of layer comes from embedded input.
        # After that, output of one layer is fed into the next layer.
        for layer in self.layers:
            embedded = layer(embedded, pad_masked)

        return embedded


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # The first sub-layer of Decoder. Its a multi-head self-attention.
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.self_attention_norm = nn.LayerNorm(d_model)
        # The second sub-layer of Decoder. Its an encoder-decoder multi-head self-attention.
        self.codec_attention= MultiHeadAttention(d_model, n_heads)
        self.codec_attention_norm = nn.LayerNorm(d_model)
        # The third sub-layer of Encoder. Its a positionwise fully connected feed-forward network.
        self.ffn_layer = PoswiseFeedForwardNet(d_model, ffn_dim)
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, source, pad_masked):
        # As we are feeding same input as query, key & value, its called self-attention.
        # target: (batch_size, seq_length, d_model)
        # pad_masked: (batch_size, seq_length, seq_length)
        output_from_attn = self.self_attention(target, target, target, pad_masked)
        output_from_attn = self.dropout(output_from_attn)
        output_from_attn = self.self_attention_norm(target + output_from_attn)
        # output_from_attn
        # output_from_attn : (batch_size, seq_length, d_model)
        # source : (batch_size, seq_length, d_model)
        output_from_codec_attn= self.codec_attention(output_from_attn, source, source, None)
        output_from_codec_attn = self.dropout(output_from_codec_attn)
        output_from_codec_attn = self.codec_attention_norm(output_from_attn + output_from_codec_attn)
        # output_from_codec_attn
        output_ffn = self.ffn_layer(output_from_codec_attn)
        output_ffn = self.dropout(output_ffn)
        output_ffn = self.ffn_layer_norm(output_from_codec_attn + output_ffn)
        # output_ffn

        return output_ffn

# Decoder is responsible to generate
# a target sentence from a context state
class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, n_layers, n_heads, ffn_dim, source_pad_idx, target_pad_idx, device, dropout=0.1):
        super(Decoder, self).__init__()

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device
        
        # It's like Word2Vec
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        # This is used to add position data with embedded words.
        self.position_embedding = PositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)

        # Series of similar layers.
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)])
        # This is to project output
        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, target, source_output):
        # This method will add masking to padded position (<pad>).
        # target: (batch_size, seq_length)
        target_mask = get_padding_mask(target, target, self.target_pad_idx)
        # This method will add masking to subsequent position.
        subsequent_mask = get_subsequent_mask(target, self.device)
        target_mask = target_mask | subsequent_mask

        embedded = self.embedding(target)
        # embedded: (batch_size, seq_length, embedding_dim)
        embedded = self.position_embedding(embedded)
        # embedded: (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)

        # The first input to the series of layer comes from embedded input.
        # After that, output of one layer is fed into the next layer. 
        for layer in self.layers:
            embedded = layer(embedded, source_output, target_mask)

        # This is to project output
        output = self.fc_out(embedded)

        return output


# Seq2Seq combines Encoder & Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        # source: (batch_size, seq_length)
        # target: (batch_size, seq_length)
        encoded = self.encoder(source)
        # encoded: (batch_size, seq_length, hidden_dim)
        output = self.decoder(target, encoded)
        # output: (batch_size, seq_length, target_vocab_size)

        return output
