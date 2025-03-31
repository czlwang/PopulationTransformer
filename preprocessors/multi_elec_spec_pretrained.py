from .stft import STFTPreprocessor
from .superlet_preprocessor import SuperletPreprocessor
import torch
import torch.nn as nn
import models
import os
import numpy as np

#This preprocssor combines a spectrogram preprocessor with a feature extracter (transformer)

def build_preprocessor(spec_name, preprocessor_cfg):
    if spec_name == "stft":
        extracter = STFTPreprocessor(preprocessor_cfg)
    elif spec_name == "superlet":
        extracter = SuperletPreprocessor(preprocessor_cfg)
    return extracter

class MultiElecSpecPretrained(nn.Module):
    def __init__(self, cfg):
        super(MultiElecSpecPretrained, self).__init__()
        self.spec_preprocessor = build_preprocessor(cfg.spec_name, cfg)

        self.cfg = cfg
        ckpt_path = cfg.upstream_ckpt
        init_state = torch.load(ckpt_path, weights_only=False)
        upstream_cfg = init_state["model_cfg"]
        self.upstream = models.build_model(upstream_cfg)
        states = init_state["model"]
        self.upstream.load_weights(states)
        self.upstream.to('cuda')#TODO hardcode

    def get_upstream_embed(self, inputs, pad_mask):
        '''
            inputs is [batch, n_electrodes, n_time, n_freq_channels]
            pad_mask is [1, n_time] 
        '''
        rep_from_layer = -1
        if "rep_from_layer" in self.cfg:
            rep_from_layer = self.cfg.rep_from_layer 
        output = self.upstream(inputs, pad_mask, intermediate_rep=True, rep_from_layer=rep_from_layer)
        return output
 
    def forward(self, wav, spec_preprocessed=None):
        '''
            wav is [n_electrodes, n_time]
            output is [n_electrodes, n_embed]
        '''
        if spec_preprocessed is None:
            spec = self.spec_preprocessor(wav)
        else:
            spec = torch.FloatTensor(spec_preprocessed)
        #spec is [n_electrodes, time, num_freq_channels]
        n_electrodes = spec.shape[0]
        pad_mask = torch.zeros(n_electrodes, spec.shape[1], dtype=bool)
        self.upstream.eval()
        spec = spec.to('cuda')#TODO hardcode
        pad_mask = pad_mask.to('cuda')#TODO hardcode
        with torch.no_grad():
            outputs = self.get_upstream_embed(spec, pad_mask)
        #outputs is [n_elec, n_time, d]
        middle = int(outputs.shape[1]/2)
        out = outputs[:,middle-5:middle+5]

        if "pool" in self.cfg and self.cfg.pool=="max":
            out, _ = out.max(axis=1)
        elif "pool" in self.cfg and self.cfg.pool=="raw":
            out = out
        else:
            out = out.mean(axis=1)
        out = out.cpu() #TODO hardcode
        return out

