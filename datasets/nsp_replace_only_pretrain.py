import random
import os
import torch
from tqdm import tqdm as tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils import data
from datasets import register_dataset
from pathlib import Path
import logging
import csv
import json
import glob
import pandas as pd
from preprocessors import build_preprocessor

log = logging.getLogger(__name__)

@register_dataset(name="nsp_replace_only_pretrain")
class NSPReplaceOnlyPretrain(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
        super().__init__()
        self.extracter = build_preprocessor(preprocessor_cfg)

        self.cfg = cfg
        self.task_cfg = task_cfg
        data_path = cfg.data_path

        manifest_path = os.path.join(data_path, "manifest.tsv")
        assert os.path.exists(manifest_path)
        manifest = []
        with open(manifest_path) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                manifest.append((row))
        self.manifest = manifest

        label_path = os.path.join(data_path, "labels.tsv")
        assert os.path.exists(label_path)
        labels = []
        with open(label_path) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                labels.append(row[0])
        self.labels = labels

        electrodes_path = os.path.join(data_path, "all_ordered_electrodes.json")
        assert os.path.exists(electrodes_path)
        with open(electrodes_path, 'r') as f:
            ordered_electrodes = json.load(f)

        elec2absolute_id = {subj:{elec:idx for idx,elec in enumerate(elecs)} for subj,elecs in ordered_electrodes.items()}
        self.ordered_electrodes = ordered_electrodes

        localization_root = os.path.join(data_path, "localization")
        self.all_localization_dfs = {}
        for fpath in glob.glob(f'{localization_root}/*'):
            subject = os.path.split(fpath)[1].split(".")[0]
            self.all_localization_dfs[subject] = pd.read_csv(fpath)

        label2idx_dict = {}
        uniq_labels = sorted(list(set(labels)))
        for idx, l in enumerate(uniq_labels):
            label2idx_dict[l] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}

        self.absolute_id = {subj: [elec2absolute_id[subj][elec] for elec in elecs] for subj,elecs in ordered_electrodes.items()} #A map from subject to a list of indices of the sub sampled channels

        self.region2id = {}
        all_dk_regions = set()
        for subj, df in self.all_localization_dfs.items():
            dk_regions = self.all_localization_dfs[subj]["DesikanKilliany"]
            all_dk_regions.update(dk_regions)
        all_dk_regions = sorted(list(all_dk_regions))
        self.region2id = {r:i for i,r in enumerate(all_dk_regions)}
        self.id2region = {i:r for r,i in self.region2id.items()} 

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def get_output_size(self):
        pass #TODO
        #return 1 #single logit

    def __len__(self):
        return len(self.manifest)

    def label2idx(self, label):
        return self.label2idx_dict[label]
        
    def replace_tokens(self, input_x, alternate_x, subsample_idxs):
        '''
            input_x is [n_elec, d]
        '''
        n_elecs = input_x.shape[0]
        sub_select_alternate_x = alternate_x[subsample_idxs] 
        random_replace_idxs = np.random.random(n_elecs) < self.task_cfg.replace_p/2
        random_keep_idxs = np.random.random(n_elecs) < self.task_cfg.replace_p/2
        random_keep_idxs = random_keep_idxs * ~random_replace_idxs

        new_x = input_x.copy()
        new_x[random_replace_idxs] = sub_select_alternate_x[random_replace_idxs] 

        replace_labels = np.zeros(random_replace_idxs.shape)
        replace_labels[random_keep_idxs] = 2
        replace_labels[random_replace_idxs] = 1
        return new_x, replace_labels

    def __getitem__(self, idx: int):
        fpath_1, fpath_2, fpath_3, subject = self.manifest[idx]

        input_x_3 = np.load(fpath_3)
        input_x_3 = self.extracter(input_x_3)
        rand_indxs = np.arange(input_x_3.shape[0])
        np.random.shuffle(rand_indxs)
        input_x_3 = input_x_3[rand_indxs]

        input_x_1 = np.load(fpath_1)
        input_x_1 = self.extracter(input_x_1)

        channel_subsample = list(range(input_x_1.shape[0]))
        random.shuffle(channel_subsample)
        choice = random.choices([10,20,30,40,50,60,70,80,90,100])[0]
        channel_subsample = channel_subsample[:choice] #TODO hardcode percentage
        half = int(len(channel_subsample)/2)
        channel_subsample_1 = sorted(channel_subsample[:half])
        channel_subsample_2 = sorted(channel_subsample[half:])

        #channel_subsample_1 = [i for i in range(input_x_1.shape[0])]
        input_x_1 = input_x_1[channel_subsample_1]
        input_x_1, replace_labels_1 = self.replace_tokens(input_x_1, input_x_3, channel_subsample_1)
        input_x_1 = torch.FloatTensor(input_x_1)

        input_x_2 = np.load(fpath_2)
        input_x_2 = self.extracter(input_x_2)
        #channel_subsample_2 = [i for i in range(input_x_2.shape[0])]
        input_x_2 = input_x_2[channel_subsample_2]
        input_x_2, replace_labels_2 = self.replace_tokens(input_x_2, input_x_3, channel_subsample_2)
        input_x_2 = torch.FloatTensor(input_x_2)

        masked_inputs_1 = input_x_1
        masked_inputs_2 = input_x_2

        embed_dim = input_x_1.shape[-1]
        cls_token = torch.ones(1,embed_dim)

        masked_inputs = torch.cat([cls_token, masked_inputs_1, masked_inputs_2])
        cls_mask = torch.zeros(1,embed_dim)

        cls_replace_label = torch.LongTensor([0])
        replace_labels_1 = torch.LongTensor(replace_labels_1)
        replace_labels_2 = torch.LongTensor(replace_labels_2)
        replace_labels = torch.cat([cls_replace_label, replace_labels_1, replace_labels_2])

        target = torch.cat([cls_token, input_x_1, input_x_2])
        #NOTE: remember not to load to cuda here
        coords_1 = self.all_localization_dfs[subject][["L", "I", "P"]].to_numpy()
        if self.cfg.get("gaussian_blur", True):
            coords_1 = coords_1 + np.random.normal(loc=0, scale=5, size=coords_1.shape)
        coords_2 = coords_1.copy()

        coords_1 = coords_1[channel_subsample_1]
        coords_2 = coords_2[channel_subsample_2]
        coords = torch.LongTensor(np.concatenate([coords_1, coords_2]))

        seq_len_1 = input_x_1.shape[0]
        seq_len_2 = input_x_2.shape[0]
        seq_id = torch.LongTensor([0]*seq_len_1 + [1]*seq_len_2)

        regions_1 = self.all_localization_dfs[subject]['DesikanKilliany']
        regions_1 = np.array([self.region2id[r] for r in regions_1])
        regions_2 = regions_1.copy()
        regions_1 = regions_1[channel_subsample_1]
        regions_2 = regions_2[channel_subsample_2]
        regions = torch.LongTensor(np.concatenate([regions_1, regions_2]))

        if self.cfg.get("region_coords", False):
            coords = regions

        return {
                "input" : masked_inputs,
                "wav": np.zeros(3),#TODO get rid of this
                "length": 1+seq_len_1 + seq_len_2,
                "coords": coords,
                "label": self.label2idx(self.labels[idx]),
                "target": target,
                "replace_label": replace_labels,
                "seq_id": seq_id,
                "subject": subject
               }
