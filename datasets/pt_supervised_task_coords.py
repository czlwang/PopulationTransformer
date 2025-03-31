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

@register_dataset(name="pt_supervised_task_coords")
class PTSupervisedTask(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
        super().__init__()
        
        self.extracter = build_preprocessor(preprocessor_cfg)
        self.cfg = cfg

        data_path = cfg.data_path

        manifest_path = os.path.join(data_path, "manifest.tsv")
        assert os.path.exists(manifest_path)
        manifest = []
        with open(manifest_path) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                manifest.append((row[0], row[1]))
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

        localization_root = os.path.join(data_path, "localization")
        all_localization_dfs = {}
        for fpath in glob.glob(f'{localization_root}/*'):
            subject = os.path.split(fpath)[1].split(".")[0]
            all_localization_dfs[subject] = pd.read_csv(fpath)

        if "sub_sample_electrodes" in cfg:
            sub_sample_electrodes_path = cfg.sub_sample_electrodes
            with open(sub_sample_electrodes_path, 'r') as f:
                sub_sample_electrodes = json.load(f)
            ordered_electrodes, all_localization_dfs = self.make_sub_sample(ordered_electrodes, all_localization_dfs, sub_sample_electrodes)

        self.ordered_electrodes = ordered_electrodes
        self.all_localization_dfs = all_localization_dfs

        label2idx_dict = {}
        uniq_labels = set(labels)
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


    def make_sub_sample(self, ordered_electrodes, all_localization_dfs, sub_sample):
        '''
            ordered_electrodes is {<subject>: [<elec>]}
            sub_sample is {<subject>: [<elec>]} but not necessarily all the subjects
        '''
        assert set(sub_sample.keys()).issubset(set(ordered_electrodes.keys()))
        assert set(sub_sample.keys()).issubset(set(all_localization_dfs.keys()))
        new_ordered_electrodes, new_all_localization_dfs = {}, {}
        for subj, elecs in sub_sample.items():
            ordered = ordered_electrodes[subj]
            assert set(elecs).issubset(set(ordered))
            new_ordered_electrodes[subj] = [e for e in ordered if e in elecs] #makes sure order is preserved

            df = all_localization_dfs[subj]
            assert set(elecs).issubset(set(df.Electrode))
            new_all_localization_dfs[subj] = df[df.Electrode.isin(elecs)]

        return new_ordered_electrodes, new_all_localization_dfs

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def get_output_size(self):
        return 1 #single logit

    def __len__(self):
        return len(self.manifest)

    def label2idx(self, label):
        return self.label2idx_dict[label]
        
    def __getitem__(self, idx: int):
        fpath, subject = self.manifest[idx]
        input_x = np.load(fpath)
        input_x = self.extracter(input_x)
        input_x = torch.FloatTensor(input_x)
        input_x = input_x[self.absolute_id[subject],:]#sub sample the channels based on selection

        embed_dim = input_x.shape[-1]
        cls_token = torch.ones(1,embed_dim)

        input_x = torch.concatenate([cls_token,input_x])

        coords = self.all_localization_dfs[subject][["L", "I", "P"]].to_numpy()
        coords = torch.LongTensor(coords)

        seq_len = input_x.shape[0] - 1
        #seq_id = torch.LongTensor([0]*seq_len + [1]*seq_len)k
        seq_id = torch.LongTensor([0]*seq_len)

        regions = self.all_localization_dfs[subject]['DesikanKilliany']
        regions = np.array([self.region2id[r] for r in regions])
        regions = torch.LongTensor(regions)

        if self.cfg.get("region_coords", False):
            coords = regions

        return {
                "input" : input_x,
                "wav": np.zeros(3),#TODO get rid of this
                "length": 1+input_x.shape[0], 
                "coords": coords,
                "label": self.label2idx(self.labels[idx]),
                "seq_id": seq_id
               }
