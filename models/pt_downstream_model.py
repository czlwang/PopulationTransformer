from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput
from models.spec_prediction_head import SpecPredictionHead
from models import build_model

@register_model("pt_downstream_model")
class PtDownstreamModel(BaseModel):
    def __init__(self):
        super(PtDownstreamModel, self).__init__()

    def forward(self, inputs, src_key_mask, positions, rep_from_layer=-1):
        outs = self.upstream(inputs, src_key_mask, positions, intermediate_rep=True)

        h = outs[:,0,:]#
        #h = outs[:,1:,:].flatten(start_dim=1)
        #h = torch.zeros(h.shape).to(h.device)#TODO

        #uncomment this line if you're concatenating the poptfAgg and raw BrainBERT embeddings together
        #h = torch.concatenate([outs[:,1:,:].flatten(start_dim=1), inputs[:,1:,:].flatten(start_dim=1)], axis=1)
        h = self.linear_out(h)
        #h = self.linear_out_2(h)
        return h

    def build_model(self, cfg):
        self.cfg = cfg
        upstream_cfg = self.cfg.upstream_cfg
        upstream_model_path = self.cfg.upstream_path
        upstream = build_model(upstream_cfg) #Eventually, think about putting this in the preprocessor
        upstream_torch = torch.load(upstream_model_path, weights_only=False)

        random_init = self.cfg.get('random_init', False)
        if not random_init:
            upstream.load_state_dict(upstream_torch["model"])

        named_params = list(upstream.named_parameters())
        names = [x[0] for x in named_params]

        if "frozen_upstream" in cfg and cfg.frozen_upstream:
            for name, param in upstream.named_parameters():
                 print(name)
                 param.requires_grad = False

        self.upstream = upstream

        self.linear_out = nn.Linear(cfg.hidden_dim, 1)
        #self.linear_out_2 = nn.Linear(cfg.hidden_dim, 1)
        #self.linear_out = nn.Linear(5*cfg.hidden_dim, 1) #TODO

        #uncomment this line if you're concatenating the poptfAgg and raw BrainBERT embeddings together
        #self.linear_out = nn.Linear(50*cfg.input_dim+50*cfg.hidden_dim, 1) #TODO
