import math
from scipy import signal, stats#TODO remove import
import time
import os
import torch
import string
import numpy as np
import h5py
import logging
from pathlib import Path
# import numpy.typing as npt

from torch.utils import data
from .h5_data import H5Data
from .h5_data_reader import H5DataReader
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from types import SimpleNamespace

log = logging.getLogger(__name__)

class MultiElectrodeSubjectData():
    def __init__(self, cfg) -> None:
        self.selected_electrodes = cfg.electrodes
        self.cfg = cfg
        self.neural_data, self.trials, self.electrodes, self.localization_df = self.get_subj_data()

    def get_subj_data(self):
        '''
            returns:
                numpy array of words
                numpy array of shape [n_electrodes, n_words, n_samples] which holds the 
                    aligned data across all trials
        '''

        seeg_data, trials = [], []
        run_ids = self.cfg.brain_runs
        subject = self.cfg.subject

        for run_id in run_ids:
            trial_data = H5Data(subject, run_id, self.cfg)
            trials.append(trial_data)
            reader = H5DataReader(trial_data, self.cfg)

            log.info("Getting filtered data")
            seeg_trial_data = reader.get_filtered_data()

            sampling_rate = 2048
            n_samples = int(sampling_rate*self.cfg.duration)
            cutoff_len = math.floor(seeg_trial_data.shape[-1] / (n_samples)) * n_samples #how many segments should we take? 
            cutoff_len = int(cutoff_len)
            seeg_trial_data = seeg_trial_data[:,:cutoff_len]
            seeg_trial_data = seeg_trial_data.reshape([seeg_trial_data.shape[0],-1, n_samples]) #NOTE hardcode
            seeg_data.append(seeg_trial_data)
        electrodes = reader.selected_electrodes
        localization_df = trial_data.get_brain_region_localization_df()

        seeg_data = np.concatenate(seeg_data, axis=1)
        return seeg_data, trials, electrodes, localization_df

