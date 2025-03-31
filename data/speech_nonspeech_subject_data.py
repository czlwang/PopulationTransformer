from scipy import signal, stats#TODO remove import
import psutil
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

class WordOnsetSubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.cfg = cfg
        self.words, self.neural_data, self.trials, self.electrodes, self.localization_df = self.get_subj_data(cfg.subject)
        #NOTE: self.electrodes (or reader.electrodes) should be the ground truth. Do not use data_cfg.electrodes


    def get_subj_data(self, subject):
        words, seeg_data, trials = [], [], []
        cached_transcript_aligns = self.cfg.cached_transcript_aligns
        for trial in self.cfg.brain_runs:
            trial_cfg = self.cfg.copy()
            if cached_transcript_aligns: #TODO: I want to make this automatic
                cached_transcript_aligns = os.path.join(cached_transcript_aligns, subject, trial)
                os.makedirs(cached_transcript_aligns, exist_ok=True)
                trial_cfg.cached_transcript_aligns = cached_transcript_aligns
            trial_data = TrialData(subject, trial, self.cfg)
            reader = TrialDataReader(trial_data, self.cfg)

            duration = self.cfg.duration
            interval_duration = self.cfg.interval_duration
            seeg_trial_no_word_data, labels = reader.get_aligned_non_words_matrix(duration=duration, interval_duration=interval_duration)
            labels['movie_id'] = trial_data.movie_id
            trials.append(trial_data)
            words.append(labels)
            seeg_data.append(seeg_trial_no_word_data)

        electrodes = reader.selected_electrodes
        localization_df = trial_data.get_brain_region_localization_df()
        neural_data = np.concatenate(seeg_data, axis=1)
        labels_df = pd.concat(words) #NOTE the index will not be unique, but the location will
        #TODO: pretty sure we are missing the get_subj_data method here
        return labels_df, neural_data, trials, electrodes, localization_df

class SentenceOnsetSubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.cfg = cfg
        self.words, self.neural_data, self.trials, self.electrodes, self.localization_df = self.get_subj_data(cfg.subject)
        #NOTE: self.electrodes (or reader.electrodes) should be the ground truth. Do not use data_cfg.electrodes

    def get_subj_data(self, subject):
        words, seeg_data, trials = [], [], []
        cached_transcript_aligns = self.cfg.cached_transcript_aligns
        for trial in self.cfg.brain_runs:
            if cached_transcript_aligns: #TODO: I want to make this automatic
                cached_transcript_aligns = os.path.join(cached_transcript_aligns, subject, trial)
                os.makedirs(cached_transcript_aligns, exist_ok=True)
                self.cfg.cached_transcript_aligns = cached_transcript_aligns
            trial_data = TrialData(subject, trial, self.cfg)
            reader = TrialDataReader(trial_data, self.cfg)

            duration = self.cfg.duration
            delta = self.cfg.delta
            interval_duration = self.cfg.interval_duration
            seeg_trial_no_word_data, labels = reader.get_aligned_speech_onset_matrix(duration=duration, interval_duration=interval_duration)
            labels['movie_id'] = trial_data.movie_id
            trials.append(trial_data)
            words.append(labels)
            seeg_data.append(seeg_trial_no_word_data)

        electrodes = reader.selected_electrodes
        localization_df = trial_data.get_brain_region_localization_df()
        neural_data = np.concatenate(seeg_data, axis=1)
        labels_df = pd.concat(words) #NOTE the index will not be unique, but the location will
        #TODO: pretty sure we are missing the get_subj_data method here
        return labels_df, neural_data, trials, electrodes, localization_df
