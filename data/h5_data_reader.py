from scipy.signal import hilbert, chirp
from tqdm import tqdm
import os
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import omegaconf
import logging
import json
# import numpy.typing as npt

from typing import Optional, List, Tuple
from scipy import signal, stats
from .utils import compute_m5_hash, stem_electrode_name

log = logging.getLogger(__name__)

class H5DataReader:
    def __init__(self, trial_data, cfg) -> None:
        '''
            Input: trial_data=ecog and word data to perform processing on
        '''
        self.freqs_to_filter = [60, 120, 180, 240, 300, 360]

        self.trial_data = trial_data
        self.cfg = cfg
        self.selected_electrodes = self.read_cfg_electrodes()

        if self.cfg.rereference=="laplacian":
            self.adj_electrodes = self.get_all_adj_electrodes()
        else:
            assert self.cfg.rereference=="None"

    def read_cfg_electrodes(self):
        electrodes = []
        if isinstance(self.cfg.electrodes, omegaconf.listconfig.ListConfig): #In this case we are given a list of electrodes
            electrodes = self.get_ordered_electrodes(self.cfg.electrodes) 
        elif isinstance(self.cfg.electrodes, str): #In this case, we are given a path to electrodes
            log.info(f"Loading electrodes from {self.cfg.electrodes}")
            #the json should have this format {<subject_name>: [electrodes]}
            with open(self.cfg.electrodes, "r") as f:
                all_electrodes = json.load(f)
            electrodes = all_electrodes[self.cfg.subject]
            electrodes = self.get_ordered_electrodes(electrodes) 
        return electrodes

    def get_adj_electrodes(self, name):
        labels = self.trial_data.get_brain_region_localization()
        all_electrode_stems = [stem_electrode_name(l) for l in labels]

        stem, num = stem_electrode_name(name)
        same_wire = [(s,n) for (s,n) in all_electrode_stems  if s==stem]
        nbrs = [(stem, num+1), (stem, num-1)]
        nbrs = [n for n in nbrs if n in all_electrode_stems]
        assert len(nbrs)==2
        return [e+str(s) for (e,s) in nbrs]

    def get_all_adj_electrodes(self):
        nbrs = [self.get_adj_electrodes(n) for n in self.selected_electrodes] #TODO debug
        flat_nbrs = [x for y in nbrs for x in y]
        return list(set(flat_nbrs))
        
    def notch_filter(self, data, freq, Q=30) -> np.ndarray:
        samp_frequency = self.trial_data.samp_frequency
        w0 = freq / (samp_frequency / 2)
        b, a = signal.iirnotch(w0, Q)
        y = signal.lfilter(b, a, data, axis = 1)
        return y

    def highpass_filter(self, data, freq, Q=30) -> np.ndarray:
        samp_frequency = self.trial_data.samp_frequency
        sos = signal.butter(Q, freq, 'highpass', fs=samp_frequency, output='sos')
        y = signal.sosfilt(sos, data, axis = 1)
        return y

    def band_filter(self, data, freqs, Q=30) -> np.ndarray:
        samp_frequency = self.trial_data.samp_frequency
        sos = signal.butter(Q, freqs, 'bandpass', analog=False, fs=samp_frequency, output='sos')
        y = signal.sosfilt(sos, data, axis = 1)
        return y

    def car_rereference(self, data_arr):
        all_ordered_labels = self.trial_data.get_brain_region_localization()
        selected = [(i,e) for i,e in enumerate(all_ordered_labels) if e in self.selected_electrodes]
        sel_idxs, sel_labels = zip(*selected)

        all_data = self.select_electrodes(all_ordered_labels)
        reref = data_arr - np.mean(all_data, axis=0)
        return reref

    def laplacian_rereference(self, data_arr):
        all_ordered_labels = self.trial_data.get_brain_region_localization()
        selected = [(i,e) for i,e in enumerate(all_ordered_labels) if e in self.selected_electrodes]
        sel_idxs, sel_labels = zip(*selected)

        ordered_nbrs = [e for e in all_ordered_labels if e in self.adj_electrodes]
        label2idx = {v:k for (k,v) in enumerate(ordered_nbrs)}

        adj_data_arr = self.select_electrodes(self.adj_electrodes)
        sel_nbrs = [self.get_adj_electrodes(n) for n in sel_labels]
        sel_nbrs_idxs = [[label2idx[l] for l in nbr_list] for nbr_list in sel_nbrs]
        sel_nbr_data = [[adj_data_arr[i] for i in idx_list] for idx_list in sel_nbrs_idxs]
        sel_nbr_data = np.array(sel_nbr_data)
        sel_nbr_data = self.filter_data(sel_nbr_data)

        #sel_nbr_data is [n_electrodes, 2, n_samples]
        laplacian = data_arr - np.mean(sel_nbr_data, axis=1)
        return laplacian
        
    def filter_data(self, data_arr):
        for f in self.freqs_to_filter:
            data_arr = self.notch_filter(data_arr, f)
        if self.cfg.high_gamma:
            band_data_arr = self.band_filter(data_arr, [70,250], Q=5)
            data_arr = band_data_arr
        return data_arr

    def de_spike(self, y):
        fuzz = 125 #number of samples around the spike to subtract out
        scaling_factor = 0.95 #how much of the spike to remove

        zscored = scipy.stats.zscore(y)
        mask = (np.abs(zscored)>4)
        fuzzed_mask = np.sign(np.convolve(mask, np.ones(fuzz), mode='same'))
        de_spiked = y - (fuzzed_mask*y*scaling_factor)
        return de_spiked

    def get_ordered_electrodes(self, selected):
        labels = self.trial_data.get_brain_region_localization()
        for e in selected:
           assert e in labels
        re_ordered_electrodes = [e for i,e in enumerate(labels) if e in selected]
        return re_ordered_electrodes
 
    def get_filtered_data(self) -> np.ndarray:
        '''
            filters out freqs from the trial data
        '''
        data_arr = self.select_electrodes(self.selected_electrodes)

        data_arr = self.filter_data(data_arr)
        if self.cfg.rereference == "CAR":
           data_arr = self.car_rereference(data_arr)
        if self.cfg.rereference=="laplacian":
           data_arr = self.laplacian_rereference(data_arr)
        if self.cfg.normalization=="zscore":
            data_arr = scipy.stats.zscore(data_arr, axis=1)
        if self.cfg.normalization=="standard":
            #This should be the same as above in principle
            mean = np.mean(data_arr, axis=1)
            std_dev = np.std(data_arr, axis=1)
            eps = 0.0001
            standardized = (data_arr - mean)/(std_dev + eps)
        if self.cfg.despike:
            for i in range(data_arr.shape[0]):
                data_arr[i] = self.cfg.de_spike(data_arr[i])
        return data_arr

    def select_electrodes(self, selected) -> np.ndarray:
        '''
            Input:
                word_window_arr = array of shape [n_electrodes, n_words, n_samples]
                electrode_labels = list of all the electrode labels for a sample
            Output:
                word_window_arr = array of shape [n_selected_electrodes, n_words, n_samples]
                                  where the order of electrodes is the same as in self.selected_electrodes

        '''
        truncate_idx = self.cfg.get("truncate_idx", -1)
        labels = self.trial_data.get_brain_region_localization()
        for e in selected:
            assert e in labels

        indices = [i for i,e in enumerate(labels) if e in selected]

        assert len(indices) == len(selected)

        electrode_data = []
        with h5py.File(self.trial_data.neural_data_file, 'r') as hf:
            raw_data = hf['data']
            for i in indices:
                if truncate_idx > -1:
                    electrode_data.append(raw_data[f'electrode_{i}'][:truncate_idx])
                else:
                    electrode_data.append(raw_data[f'electrode_{i}'][:])
        electrode_data_arr = np.stack(electrode_data)
            
        return electrode_data_arr
