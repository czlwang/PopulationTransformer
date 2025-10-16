import random
from scipy.signal import hilbert, chirp
from tqdm import tqdm
import os
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import logging
import omegaconf

from typing import Optional, List, Tuple
from .trial_data import TrialData
from scipy import signal, stats
from .utils import compute_m5_hash
from .h5_data_reader import H5DataReader
from pathlib import Path

log = logging.getLogger(__name__)

class TrialDataReader(H5DataReader):
    def __init__(self, trial_data, cfg) -> None:
        '''
            Input: trial_data=ecog and word data to perform processing on
        '''
        super().__init__(trial_data, cfg)

        self.start_col = 'start'
        self.end_col = 'end'
        self.trig_time_col = 'movie_time'
        self.trig_idx_col = 'index'
        self.est_idx_col = 'est_idx'
        self.est_end_idx_col = 'est_end_idx'
        self.word_time_col = 'word_time'
        self.word_text_col = 'text'
        self.is_onset_col = 'is_onset'
        self.is_offset_col = 'is_offset'

        cached_transcript_aligns = cfg.cached_transcript_aligns
        self.aligned_script_df = self.get_aligned_movie_transcript(cached_transcript_aligns)

        self.selected_words = [] #TODO hardcode
        #self.selected_electrodes is set in the superclass

    def estimate_sample_index(self, t, near_t, near_trig):
        '''
            input: movie time t and the closest trigger time
            returns: linear interpolation to the nearest sample index to t
        '''
        samp_frequency = self.trial_data.samp_frequency
        trig_diff = (t-near_t)*samp_frequency
        return round(near_trig+trig_diff)

    def add_estimated_sample_index(self, w_df: pd.DataFrame) -> pd.DataFrame:
        '''
            input: a dataframe of word features
            returns: the input word dataframe, but augmented with onset and offset times
        '''
        tmp_w_df = w_df.copy(deep=True)
        trigs_df = self.trial_data.get_trigger_times()
        last_t = trigs_df.loc[len(trigs_df) - 1, self.trig_time_col]

        for i, t, endt in tqdm(zip(w_df.index, w_df[self.start_col], w_df[self.end_col])):
            if t > last_t:
                break
            idx = (abs(trigs_df[self.trig_time_col] - t)).idxmin()
            tmp_w_df.loc[i, :] = w_df.loc[i, :]
            trigger_t = trigs_df.loc[idx, self.trig_time_col]
            trigger_idx = trigs_df.loc[idx, self.trig_idx_col] 
            tmp_w_df.loc[i, self.est_idx_col] = self.estimate_sample_index(t, trigger_t, trigger_idx)

            end_idx = (abs(trigs_df[self.trig_time_col] - endt)).idxmin()
            end_trigger_t = trigs_df.loc[end_idx, self.trig_time_col]
            end_trigger_idx = trigs_df.loc[end_idx, self.trig_idx_col] 
            tmp_w_df.loc[i, self.est_end_idx_col] = self.estimate_sample_index(endt, end_trigger_t, end_trigger_idx)
        return tmp_w_df

    def get_aligned_movie_transcript(self, cached_transcript_aligns: str) -> pd.DataFrame:
        '''
            returns the dataframe of word data for the trial, but augmented with onset and offset times
        '''

        save_path = None
        if cached_transcript_aligns:
            save_path = os.path.join(cached_transcript_aligns, "aligned_script.h5") 
            computed_hash = compute_m5_hash(self.trial_data.transcript_file)

        if save_path and os.path.exists(save_path): 
            cached_df = pd.read_hdf(save_path, key='transcript_data')
            if cached_df['orig_transcript_hash'][0] == computed_hash:
                return cached_df

        words_df = self.trial_data.get_movie_transcript()
        words_df = self.add_estimated_sample_index(words_df)

        if save_path:
            words_df['orig_transcript_hash'] = computed_hash #TODO eventually look into storing metadata
            words_df.to_hdf(save_path, key='transcript_data')

        return words_df

    def select_words(self, words_df: pd.DataFrame, index_subsample=None) -> pd.DataFrame:
        '''
        Input:
            word_window_arr = array of shape [n_electrodes, n_words, n_samples]
            words_df = pandas dataframe of word features 
        Output:
            pandas dataframe of the word data where only
                rows with the selected words are present
            
        '''
        filtered_df = words_df
        if index_subsample is not None:
            filtered_df = words_df.loc[index_subsample]
        filtered_df = filtered_df[filtered_df['text'] != '']

        if self.selected_words==[]:
            return filtered_df

        filtered_df = filtered_df[filtered_df['text'].isin(self.selected_words)]
        return filtered_df

    def make_cached_data_array_file_name(self, electrode, subject_data_type):
        cfg = self.cfg
        return f'cache_{cfg.duration}_d_{cfg.delta}_del_{cfg.rereference}_{cfg.subject}_{electrode}_{self.trial_data.trial_id}_{subject_data_type}_{cfg.high_gamma}_hg'
  
    def make_cached_data_array_file_name_for_segment(self, cache_dir, electrode):
        """
        Given a cache_dir (or file name) and an electrode, returns a new cache name
        with the electrode replaced, keeping all other parts the same.
        Assumes naming convention:
        cache_{duration}_d_{delta}_del_{rereference}_{subject}_{electrode}_{trial_id}_{subject_data_type}_{high_gamma}_hg
        """
        import os
        base_name = os.path.basename(cache_dir)
        parts = base_name.split('_')

        try:
            sub_idx = parts.index('sub')
            electrode_idx = sub_idx + 2  # subject number is sub_idx+1, electrode is sub_idx+2
            parts[electrode_idx] = str(electrode)
        except (ValueError, IndexError):
            raise ValueError(f"Cache dir name {cache_dir} does not match expected format.")

        new_name = '_'.join(parts)
        return new_name
  
    def save_cache(self, cached_data_path, results, labels_df, subject_data_type):
        for i,e in enumerate(self.selected_electrodes):
            arr = np.expand_dims(results[i], axis=0)
            data_dir = self.make_cached_data_array_file_name(e, subject_data_type)
            e_path = os.path.join(cached_data_path, data_dir)
            Path(e_path).mkdir(exist_ok=True, parents=True)
            data_array_path = os.path.join(e_path, "array.npy")
            np.save(data_array_path, arr)
            data_label_path = os.path.join(e_path, "labels.csv")
            labels_df.to_csv(data_label_path)
            
    def find_larger_interval_cache(self, cached_data_path, subject_data_type, desired_duration, desired_delta):
        """
        Search for a cache with the same parameters but a larger duration and a delta that allows slicing.
        Returns (duration, cache_dir) if found, else (None, None).
        Ensures all parameters except duration and delta match by using make_cached_data_array_file_name.
        """
        found = None
        found_duration = None
        found_delta = None
        # Get the current cache name and split by '_', replacing duration and delta with wildcards
        example_name = self.make_cached_data_array_file_name(self.selected_electrodes[0], subject_data_type)
        example_parts = example_name.split('_')
        # duration is always at index 1, delta at index 3
        for d in os.listdir(cached_data_path):
            if not d.startswith('cache_'):
                continue
            parts = d.split('_')
            if len(parts) != len(example_parts):
                continue
            try:
                dur = float(parts[1])
                delta = float(parts[3])
            except Exception:
                continue
            if dur < desired_duration:
                continue
            # Only allow if the cached delta is <= desired_delta (so we can slice)
            if delta > desired_delta:
                continue
            # Check all other parts except duration (index 1) and delta (index 3)
            match = all(parts[i] == example_parts[i] for i in range(len(parts)) if i not in [1, 3])
            if match:
                # Prefer the smallest duration that is still >= desired_duration and largest delta <= desired_delta
                if (found is None or dur < found_duration or (dur == found_duration and delta > found_delta)):
                    found = d
                    found_duration = dur
                    found_delta = delta
        if found:
            return found_duration, os.path.join(cached_data_path, found)
        return None, None
    
    def extract_smaller_interval_from_cache(self, arr, labels, large_duration, small_duration, samp_frequency, align='center'):
        n_samples_large = arr.shape[-1]
        n_samples_small = int(small_duration * samp_frequency)
        if n_samples_small > n_samples_large:
            raise ValueError("Requested interval is larger than cached interval")
        if align == 'center':
            start = (n_samples_large - n_samples_small) // 2
        elif align == 'left':
            start = 0
        elif align == 'right':
            start = n_samples_large - n_samples_small
        else:
            raise ValueError("align must be 'center', 'left', or 'right'")
        end = start + n_samples_small
        arr_small = arr[..., start:end]
        return arr_small, labels
    

    def load_from_cache(self, cached_data_path, subject_data_type):        
        all_labels = None
        arr_list = []
        total_rows = 0

        # First pass: validate and get total number of rows
        for e_name in self.selected_electrodes:
            data_dir = self.make_cached_data_array_file_name(e_name, subject_data_type)

            data_label_path = os.path.join(cached_data_path, data_dir, "labels.csv")
            data_array_path = os.path.join(cached_data_path, data_dir, "array.npy")

            if not (os.path.exists(data_label_path) and os.path.exists(data_array_path)):
                return None, None

            # Load labels once
            labels = pd.read_csv(data_label_path)
            if all_labels is None:
                all_labels = labels
            else:
                assert all(labels == all_labels)

            # Open array in mmap mode (doesn't load into RAM yet)
            arr = np.load(data_array_path, mmap_mode='r')
            arr_list.append(arr)
            total_rows += arr.shape[0]

        # Preallocate final array
        example_shape = arr_list[0].shape[1:]  # e.g., (n_samples,)
        final_shape = (total_rows, *example_shape)
        combined_arr = np.empty(final_shape, dtype=np.float32)

        # Second pass: fill final array in chunks
        curr_idx = 0
        for arr in arr_list:
            n = arr.shape[0]
            combined_arr[curr_idx:curr_idx+n] = arr
            curr_idx += n

        return combined_arr, all_labels
 

    def load_segment_from_cache(self, cached_data_path, data_dir_large, duration, align='center'):
        """
        Loads a cached array with a larger interval, slices it to the desired interval_duration,
        and returns the combined array and labels. Uses extract_smaller_interval_from_cache as helper.
        Ensures the returned data shape and labels index match.
        """
        all_labels = None
        arr_list = []
        total_rows = 0
        samp_frequency = self.trial_data.samp_frequency

        for e_name in self.selected_electrodes:
            data_dir = self.make_cached_data_array_file_name_for_segment(data_dir_large, e_name)
            
            data_label_path = os.path.join(cached_data_path, data_dir, "labels.csv")
            data_array_path = os.path.join(cached_data_path, data_dir, "array.npy")

            if not (os.path.exists(data_label_path) and os.path.exists(data_array_path)):
                return None, None

            labels = pd.read_csv(data_label_path)
            if all_labels is None:
                all_labels = labels
            else:
                assert all(labels == all_labels)

            arr = np.load(data_array_path, mmap_mode='r')
            large_duration = arr.shape[-1] / samp_frequency
            arr_small, _ = self.extract_smaller_interval_from_cache(
                arr, labels, large_duration, duration, samp_frequency, align=align
            )
            arr_list.append(arr_small)
            total_rows += arr_small.shape[0]

        example_shape = arr_list[0].shape[1:]
        final_shape = (total_rows, *example_shape)
        combined_arr = np.empty(final_shape, dtype=np.float32)

        curr_idx = 0
        for arr in arr_list:
            n = arr.shape[0]
            combined_arr[curr_idx:curr_idx+n] = arr
            curr_idx += n

        return combined_arr, all_labels

    def get_aligned_linguistic_control_matrix(self, duration: int=3, interval_duration=None, onsets_only=True):
        '''
            input:
                duration=context to return as input in seconds. For example, if duration=5s, then 
                    5s worth of data will be returned
                interval_duration=intervals to divide the movie into. For example, if
                    interval_duration=1s, the movie will be divided into 1s chunks, and each chunk
                    will be returned embedded in 5s worth of context.
            output:
                an array of shape [n_electrodes, n_words, n_samples]
                for each electrode, each row is the ecog data from self.trial_data for a given word
        '''
        if interval_duration is None:
            interval_duration = duration
        log.info("Getting onset and non-speech intervals")
        cached_data_path = self.cfg.get("cached_data_array", None)
        cache_name = "onset_finetuning" if onsets_only else "speech_finetuning"
        reload_caches = self.cfg.get("reload_caches", False)
        if cached_data_path is not None and not reload_caches:
            arr, labels = self.load_from_cache(cached_data_path, cache_name)
            if labels is not None and arr is not None:
                return arr, labels
            
            #here we can also implement a check to load the data from cache if there exists same params and smaller interval
            dur, cache_dir_large = self.find_larger_interval_cache(cached_data_path, cache_name, duration, self.cfg.delta)
            log.info(f"loaded intervals from cached data {cached_data_path}")
            if dur is not None and cache_dir_large is not None:   
                arr, labels = self.load_segment_from_cache(cached_data_path, cache_dir_large, duration)
                if labels is not None and arr is not None:
                    log.info(f"using intervals from cached data array")
                    return arr, labels
            ###########
        

        filtered_data = self.get_filtered_data()

        w_df = self.aligned_script_df
        w_df = self.select_words(w_df)

        samp_frequency = self.trial_data.samp_frequency
        input_window_duration = int(duration*samp_frequency)
        input_left, input_right = int(input_window_duration/2), input_window_duration-int(input_window_duration/2) 

        interval_window_duration = int(interval_duration*samp_frequency)

        w_df = w_df.iloc[w_df[self.est_idx_col].dropna().index] #drop all rows that don't have a start time
#NOTE: key assumption is that the order of the electrodes in the argument list is the same as the order of the electrodes in the data array. #TODO check if this is true
        start_idxs = w_df[self.est_idx_col].astype(int).tolist()
        end_idxs = w_df[self.est_end_idx_col].astype(int).tolist()
        word_intervals = list(zip(start_idxs, end_idxs))
        total_length = filtered_data.shape[-1]
        total_n_intervals = int(total_length/interval_window_duration)
        all_intervals = [(i*interval_window_duration, (i+1)*interval_window_duration) for i in range(total_n_intervals)]
        intersect = lambda a, b: a[0] < b[1] and a[1] > b[0]
        intersect_with_word = lambda a: np.any([intersect(a,b) for b in word_intervals])
        non_word_intervals = list(filter(lambda x: not intersect_with_word(x), all_intervals))

        if onsets_only:
            w_df = w_df[w_df.is_onset.astype(bool)]

        start_idxs = w_df[self.est_idx_col].astype(int)
        start_idxs = (start_idxs - input_left).astype(int)
        end_idxs = (start_idxs + input_window_duration).astype(int)
        word_intervals = list(zip(start_idxs.tolist(), end_idxs.tolist()))

        valid_non_word_intervals = []
        for (start, end) in non_word_intervals:
            center = int((start+end)/2)
            if center-input_window_duration>0 and center+input_window_duration<filtered_data.shape[1]:
                valid_non_word_intervals.append((center-input_left, center+input_right))

        all_non_word_samples = np.stack([filtered_data[:, start:end] for (start,end) in valid_non_word_intervals])
        all_word_samples = np.stack([filtered_data[:, start:end] for (start,end) in word_intervals])

        balanced_len = min(len(all_word_samples), len(all_non_word_samples))
        random.seed(42)
        non_word_idxs = list(range(len(all_non_word_samples)))
        non_word_idxs = random.sample(non_word_idxs, balanced_len)
        all_non_word_samples = all_non_word_samples[non_word_idxs]
        word_idxs = list(range(len(all_word_samples)))
        word_idxs = random.sample(word_idxs, balanced_len)
        all_word_samples = all_word_samples[word_idxs]

        word_times = np.array([x[0] for x in word_intervals])
        all_word_times = word_times[word_idxs]
        non_word_times = np.array([x[0] for x in valid_non_word_intervals])
        all_non_word_times = non_word_times[non_word_idxs]

        result = np.concatenate([all_word_samples, all_non_word_samples], axis=0)
        times = np.concatenate([all_word_times, all_non_word_times], axis=0)
        labels = np.repeat([True, False], [all_word_samples.shape[0], all_non_word_samples.shape[0]])
        result = np.transpose(result, [1,0,2]) #[n_electrodes, n_intervals, n_samples]
        labels_df = pd.DataFrame({"linguistic_content": labels,
                                  "times": times})

        if cached_data_path is not None:
            self.save_cache(cached_data_path, result, labels_df, cache_name)
        return result, labels_df

    def get_aligned_non_words_matrix(self, duration: int=3, interval_duration=None):
        return self.get_aligned_linguistic_control_matrix(duration, interval_duration=interval_duration, onsets_only=False)

    def get_aligned_speech_onset_matrix(self, duration: int=3, interval_duration=None):
        return self.get_aligned_linguistic_control_matrix(duration, interval_duration=interval_duration, onsets_only=True)

    def get_aligned_predictor_matrix(self, duration: int=3, delta: int=-1, index_subsample=None, save_path: Optional[str]=None) -> Tuple[pd.DataFrame, np.ndarray]:
        '''
            input:
                delta=context to take before onset
                duration=context to take after start. Note that the start = onset + delta.
                save_path=where to save/load the aligned matrix
            output:
                an array of shape [n_electrodes, n_words, n_samples]
                for each electrode, each row is the ecog data from self.trial_data for a given word
        '''
        cached_data_path = self.cfg.get("cached_data_array", None)
        reload_caches = self.cfg.get("reload_caches", False)
        if cached_data_path is not None and not reload_caches:
            log.info(f"cached_data_path {cached_data_path}")
            try:
                arr, labels = self.load_from_cache(cached_data_path, "subject-data")
                log.info(f"loaded cached data {cached_data_path}")
            except:
                import pdb; pdb.set_trace()
            if labels is not None and arr is not None:
                log.info(f"using cached data array")
                return labels, arr
            
            #here we can also implement a check to load the data from cache if there exists same params and smaller interval
            dur, cache_dir_large = self.find_larger_interval_cache(cached_data_path, "subject-data", duration, delta)
            log.info(f"loaded intervals from cached data {cached_data_path}")
            if dur is not None and cache_dir_large is not None:   
                arr, labels = self.load_segment_from_cache(cached_data_path, cache_dir_large, duration)
                if labels is not None and arr is not None:
                    log.info(f"using intervals from cached data array")
                    return labels, arr
            ###########

        filtered_data = self.get_filtered_data()

        w_df = self.aligned_script_df
        w_df = self.select_words(w_df, index_subsample=index_subsample)
        samp_frequency = self.trial_data.samp_frequency
        window_duration = int(duration*samp_frequency)
        window_onset = int(delta*samp_frequency)

        w_df = w_df.loc[w_df[self.est_idx_col].dropna().index] #drop all rows that don't have a start time
        word_window_arr = np.empty((filtered_data.shape[0], len(w_df.index), window_duration))

        log.info('Generating aligned predictor matrix of size {}'.format((filtered_data.shape[0], len(w_df.index), window_duration)))
        start_idxs = w_df[self.est_idx_col].astype(int) + window_onset
        end_idxs = start_idxs + window_duration
        for i,word_idx in tqdm(enumerate(w_df.index)):
            row = w_df.loc[word_idx]
            try:
                start = start_idxs[word_idx]
                end = end_idxs[word_idx]
                word_window_arr[:, i, :] = filtered_data[:, start:end]
            except ValueError as err:
                print('Neural recording stopped before movie ended')
                break
        w_df['ecog_idx'] = range(word_window_arr.shape[1])
        if cached_data_path is not None:
            self.save_cache(cached_data_path, word_window_arr, w_df, "subject-data")
        return w_df, word_window_arr #TODO check that this w_df is the one being used at train time  
