from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput
from models.spec_prediction_head import SpecPredictionHead

@register_model("pt_model_custom")
class PtModelCustom(BaseModel):
    def __init__(self):
        super(PtModelCustom, self).__init__()

    def forward(self, inputs, src_key_mask, positions, intermediate_rep=False, rep_from_layer=-1):
        input_specs, pos = self.input_encoding(inputs, positions)
        input_specs = input_specs.transpose(0,1) #nn.Transformer wants [seq, batch, dim]
        output_specs, weights = self.transformer_encoder(input_specs, src_key_padding_mask=src_key_mask) 
        output_specs = output_specs.transpose(0,1) #[batch, seq, dim]
        
        ## Returning intermediate representation of all tokens
        if intermediate_rep:
            if self.cfg.get("attention_weights", False): 
                # Return a tuple of length 2 if we want attention_weights
                return output_specs, weights
            else: 
                # Otherwise, only return the output specs 
                return output_specs
        
        ## Returning cls token representation
        cls_output = self.cls_head(output_specs[:,0,:])
        if self.use_token_cls_head:
            token_cls_output = self.token_cls_head(output_specs[:,1:,:])
            cls_output = (cls_output, token_cls_output)

        output_specs = self.spec_prediction_head(output_specs)
        
        if self.cfg.get("attention_weights", False): 
            return output_specs, weights
        else: 
            return output_specs, cls_output

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.bias.data.fill_(1.0)

    def build_model(self, cfg):
        self.cfg = cfg
        self.input_encoding = TransformerEncoderInput(cfg, dropout=0.1)
        encoder_layer = TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.n_head, activation=cfg.layer_activation, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.spec_prediction_head = SpecPredictionHead(cfg)
        self.apply(self.init_weights)

        cls_out_dim = 1
        if "cls_out_dim" in self.cfg:
            cls_out_dim = self.cfg.cls_out_dim
        self.cls_head = nn.Linear(self.cfg.hidden_dim, cls_out_dim)

        self.use_token_cls_head = False
        if 'use_token_cls_head' in self.cfg and self.cfg.use_token_cls_head:
            self.token_cls_head = nn.Linear(self.cfg.hidden_dim, cls_out_dim)
            self.use_token_cls_head = True



## A simpler implementation of the nnTransformer that allows for attention weights to be returned
# From https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/  
import torch.nn.functional as F
import copy


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, average_attn_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights
