from sympy import Mul

import torch 
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # shape -> ([4, 1])
        # a^b = e^(b*ln(a))
        # div_term has 256 values, which are used for the sine and cosine functions.
        # Sine values fill the even positions and cosine values fill the odd positions,
        # resulting in 512 columns in pe.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape -> torch.Size([256])
        # position * div_term:
        # each position is multiplied by all 256 values in div_term,
        # producing 256 values for each position.
        # These values are used for sin at even indices and cos at odd indices, producing 512 for each position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # add batch dimension (1, seq_len, d_model)

        #store it inside the model as a fixed tensor that moves/saves with the model but isn't trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # take all batches
        # take positions 0 -> 4
        # take all 512 dimensions
        # whatever the batch size of x pytorch will automatically broadcasts the positional encoding 
        # Sentence 1 embedding + positional encoding
        # Sentence 2 embedding + positional encoding
        # Sentence 3 embedding + positional encoding
        # Sentence 4 embedding + positional encoding
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)



class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps  # for numerical stability also to avoid division by zero 
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied (learnable parameter)
        self.bias = nn.Parameter(torch.zeros(1)) # Added (learnable parameter)

    def forward(self, x):
        # dim=-1 (last dimension = embedding dimension)
        # for each token, calculate the mean and std of its embedding values
        # keepdim -> to keep the dimension that we calculated over (dimensions of x)
        # if we have 4 tokens and 100 embedding the dimensions of mean and std will be (4,1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttention(nn.Module):

    def __init__(self, h: int, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # replace the attention score with -1e9 wherever the mask is 0

        # apply softmax to the last dimension (for each row ) because we want each token's attention score over all tokens to become probabilities
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores
    


    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> after transpose (batch, h, seq_len, d_k)
        # we want each head to watch seq and d_k  -> each head see the full sentence
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 

        x, self.attention_scores = MultiHeadAttention.attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)

        # (batch, h, seq_len, d_k)  -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        # -1 here let pytorch to figure out this dimension automatically instead of writing seq_len manually
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)



class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, layer):
        # prenorm implementation
        # x -> normlayer -> layer -> dropout -> + x ---> output
        # the original transformer paper uses
        # x -> layer -> dropout -> + x -> normlayer ---> output
        # self.norm(self.dropout(layer(x)) + x)
        return x + self.dropout(layer(self.norm(x)))



class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork, dropout: float):
            super().__init__()
            self.self_attention_block = self_attention_block
            self.feed_forward_block = feed_forward_block
            self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])

    # src_mask because we want to hide the interaction of the padding word with other words
    def forward(self, x, src_mask):
        # use lambda to delay the call because residual connection expects a function not the result of calling it
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x



class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class DecoderBlock(nn.Module):

    def __init__(
            self, 
            self_attention_block: MultiHeadAttention, 
            cross_attention_block: MultiHeadAttention, 
            feed_forward_block: FeedForwardNetwork, 
            dropout: float
        ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(3)])

    def forward(self,x, encoder_output, src_mask, tgt_mask):
        # mask of the decoder because this is the self attention of the decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x



class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

# it projects the decoder's hidden representation into another space (vocab space)
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)



class Transformer(nn.Module):
    def __init__(
            self, 
            encoder: Encoder,
            decoder: Decoder,
            src_embed: InputEmbeddings,
            tgt_embed: InputEmbeddings,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer
        ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)



def build_transformer(
        src_vocab_size: int, 
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048
    ) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(h, d_model, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(layers=nn.ModuleList(encoder_blocks))

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(h, d_model, dropout)
        cross_attention_block = MultiHeadAttention(h, d_model, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    decoder = Decoder(layers=nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer