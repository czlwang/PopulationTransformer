import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        '''
        From https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2
        '''
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq):
        #seq is [batch, len, dim]
        assert len(seq.shape) == 3
        pos_enc = self.pe[:,:seq.size(1),:]
        out = seq + pos_enc
        test = torch.zeros_like(seq) + pos_enc
        return out, pos_enc

class MultiSubjBrainPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(MultiSubjBrainPositionalEncoding, self).__init__()

        assert d_model%4==0
        pe_dim = int(d_model/4) #The idea is that each one of the XYZ + seq id will get their own position embedding
        pe = torch.zeros(max_len, pe_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, seq, positions): 
        #seq is [batch, len, dim]
        assert len(seq.shape) == 3
        coords, seq_id = positions

        #self.pe is [1, max_len, d] size
        #coords is [batch, seq_len-1, 3] size
        p_embed = self.pe[0,coords]#2 axis is the XYZ axis
        n_batch, seq_len, n_axes, d_p_embed = p_embed.shape
        p_embed = p_embed.reshape(n_batch, seq_len, n_axes*d_p_embed)#flatten the last two dims into one position vector
        seq_id = self.pe[0,seq_id]
        input_embeddings = torch.cat([p_embed, seq_id], axis=-1)

        batch_size, _, d_embed = seq.shape
        cls_embed = torch.unsqueeze(self.pe[0,0].repeat(batch_size,4), 1) #[batch_size, 1, d]
        
        input_embeddings = torch.cat([cls_embed, input_embeddings], axis=1)#[batch, seq_len, d]

        out = seq + input_embeddings
        return out, input_embeddings

class TransformerEncoderInput(nn.Module):
    def __init__(self, cfg, dropout=0.1):
        super(TransformerEncoderInput, self).__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim)
        if "position_encoding" in self.cfg and self.cfg.position_encoding == "multi_subj_position_encoding":
            self.positional_encoding = MultiSubjBrainPositionalEncoding(self.cfg.hidden_dim)
        else:
            print("Default encoding")
            self.positional_encoding = PositionalEncoding(self.cfg.hidden_dim)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_specs, positions=None):
        input_specs = self.in_proj(input_specs)
        if isinstance(self.positional_encoding, PositionalEncoding):
            input_specs, pos_enc = self.positional_encoding(input_specs)
        else:
            input_specs, pos_enc = self.positional_encoding(input_specs, positions=positions)
        input_specs = self.layer_norm(input_specs)
        input_specs = self.dropout(input_specs)
        return input_specs, pos_enc
