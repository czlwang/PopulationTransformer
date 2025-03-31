from scipy import signal, stats#TODO remove import
import time
import os
import torch
import string
import numpy as np
import h5py
# import numpy.typing as npt

from torch.utils import data
from .trial_data import TrialData
from .trial_data_reader import TrialDataReader
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from types import SimpleNamespace

class SubjectData():
    def __init__(self, cfg, index_subsample=None) -> None:
        self.cfg = cfg
        self.words, self.neural_data, self.trials, self.electrodes, self.localization_df = self.get_subj_data(cfg.subject, index_subsample=index_subsample)
        self.subject = cfg.subject
        #NOTE: self.electrodes (or reader.electrodes) should be the ground truth. Do not use data_cfg.electrodes

    def get_transcript_dfs(self):
        words, seeg_data, trials = [], [], []
        cached_transcript_aligns = self.cfg.cached_transcript_aligns
        w_dfs = []
        for trial in self.cfg.brain_runs:
            trial_cfg = self.cfg.copy()
            if cached_transcript_aligns: #TODO: I want to make this automatic
                cached_transcript_aligns = os.path.join(cached_transcript_aligns, self.subject, trial)
                os.makedirs(cached_transcript_aligns, exist_ok=True)
                trial_cfg.cached_transcript_aligns = cached_transcript_aligns
            trial_data = TrialData(self.subject, trial, trial_cfg)
            reader = TrialDataReader(trial_data, trial_cfg)
            w_df = reader.aligned_script_df
            w_dfs.append(w_df)
        return w_dfs

    def get_subj_data(self, subject, index_subsample=None):
        words, seeg_data, trials = [], [], []
        cached_transcript_aligns = self.cfg.cached_transcript_aligns
        for trial in self.cfg.brain_runs:
            trial_cfg = self.cfg.copy()
            if cached_transcript_aligns: #TODO: I want to make this automatic
                cached_transcript_aligns = os.path.join(cached_transcript_aligns, subject, trial)
                os.makedirs(cached_transcript_aligns, exist_ok=True)
                trial_cfg.cached_transcript_aligns = cached_transcript_aligns
            trial_data = TrialData(subject, trial, trial_cfg)
            reader = TrialDataReader(trial_data, trial_cfg)

            trial_words, seeg_trial_data = reader.get_aligned_predictor_matrix(duration=self.cfg.duration, delta=self.cfg.delta, index_subsample=index_subsample)
            if index_subsample is None:
                assert (range(seeg_trial_data.shape[1]) == trial_words.index).all()
            trial_words['movie_id'] = trial_data.movie_id
            trials.append(trial_data)
            words.append(trial_words)
            seeg_data.append(seeg_trial_data)

        #get electrode labels
        electrodes = reader.selected_electrodes
        localization_df = trial_data.get_brain_region_localization_df()
        neural_data = np.concatenate(seeg_data, axis=1)
        #neural_data is [n_electrodes, n_words, n_samples]
        words_df = pd.concat(words) #NOTE the index will not be unique, but the location will
        return words_df, neural_data, trials, electrodes, localization_df
