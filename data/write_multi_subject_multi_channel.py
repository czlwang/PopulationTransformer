from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
from datasets import build_dataset
from preprocessors import build_preprocessor
from data.subject_data import SubjectData
from data.multi_electrode_subj_data import MultiElectrodeSubjectData
from data.speech_nonspeech_subject_data import WordOnsetSubjectData, SentenceOnsetSubjectData
import json
import numpy as np
from tqdm import tqdm as tqdm
import csv
from scipy.stats import zscore

log = logging.getLogger(__name__)

def get_spec_target_pretraining_data_and_labels(subject_data):
    seeg_data = subject_data.neural_data
    labels = [1]*seeg_data.shape[1] #number of examples #TODO put useful information here
    return seeg_data, labels

def get_pretraining_data_and_labels(subject_data):
    seeg_data = subject_data.neural_data
    #labels = [1]*seeg_data.shape[1] #number of examples #TODO put useful information here
    labels = list(range(seeg_data.shape[1]))
    return seeg_data, labels

def get_pos_features_data_and_labels(subject_data):
    word_df = subject_data.words
    assert list(word_df.index)==list(range(len(word_df)))
    seeg_data = subject_data.neural_data
    assert len(subject_data.electrodes) == seeg_data.shape[0]

    noun_idxs = list(word_df[word_df.pos=="NOUN"].index)
    verb_idxs = list(word_df[word_df.pos=="VERB"].index)
    feat_idxs = noun_idxs + verb_idxs

    feat_df = word_df.loc[feat_idxs]

    feat_df["feat_bucket"] = False
    feat_df.loc[verb_idxs, "feat_bucket"] = True

    seeg_data = seeg_data[:,feat_idxs]
    labels = np.array(feat_df["feat_bucket"])
    labels = [[label] for label in labels]
    return seeg_data, labels

def get_word_features_labels(subject_data, key):
    '''
    For tasks that have data and labels given on the word level, e.g., rms and pitch
    Note that this happens on the subject level. That is, trials are aggregated if there are multiple.
    '''
    word_df = subject_data.words

    sorted_df = word_df.sort_values(by=key)
    q25_idx = int(len(sorted_df)/4)
    q75_idx = int(3*len(sorted_df)/4)
    low_idxs = list(sorted_df[:q25_idx].index)
    high_idxs = list(sorted_df[q75_idx:].index)
    feat_idxs = low_idxs + high_idxs

    feat_df = word_df.loc[feat_idxs]

    feat_df["feat_bucket"] = False
    feat_df.loc[high_idxs, "feat_bucket"] = True

    labels = list(np.array(feat_df["feat_bucket"]))
    times = list(feat_df["start"])
    all_labels = list(zip(labels, times))
    return feat_idxs, all_labels

def get_word_features_data_and_labels(subject_data, key):
    '''
    For tasks that have data and labels given on the word level, e.g., rms and pitch
    Note that this happens on the subject level. That is, trials are aggregated if there are multiple.
    '''
    word_df = subject_data.words
    assert list(word_df.index)==list(range(len(word_df)))
    seeg_data = subject_data.neural_data
    assert len(subject_data.electrodes) == seeg_data.shape[0]

    sorted_df = word_df.sort_values(by=key)
    q25_idx = int(len(sorted_df)/4)
    q75_idx = int(3*len(sorted_df)/4)
    low_idxs = list(sorted_df[:q25_idx].index)
    high_idxs = list(sorted_df[q75_idx:].index)
    feat_idxs = low_idxs + high_idxs

    feat_df = word_df.loc[feat_idxs]

    feat_df["feat_bucket"] = False
    feat_df.loc[high_idxs, "feat_bucket"] = True

    seeg_data = seeg_data[:,feat_idxs]
    labels = list(np.array(feat_df["feat_bucket"]))
    times = list(feat_df["start"])
    all_labels = list(zip(labels, times))
    return seeg_data, all_labels

def get_reg_word_features_data_and_labels(subject_data, key):
    '''
    For tasks that have data and labels given on the word level, e.g., rms and pitch
    Note that this happens on the subject level. That is, trials are aggregated if there are multiple.
    '''
    feat_df = subject_data.words
    assert list(feat_df.index)==list(range(len(feat_df)))
    seeg_data = subject_data.neural_data
    assert len(subject_data.electrodes) == seeg_data.shape[0]

    labels = np.array(feat_df[key])

    labels = zscore(labels)
    return seeg_data, labels

def get_word_onset_data_and_labels(subject_data):
    labels = np.array(subject_data.words["linguistic_content"])
    seeg_data = subject_data.neural_data
    labels = [label for label in labels]

    times = list(subject_data.words["times"])
    all_labels = list(zip(labels, times))
    return seeg_data, all_labels

def get_pitch_data_and_labels(subject_data):
    return get_word_features_data_and_labels(subject_data, "pitch")

def get_rms_data_and_labels(subject_data):
    return get_word_features_data_and_labels(subject_data, "rms")

def get_pos_data_and_labels(subject_data):
    return get_pos_features_data_and_labels(subject_data)

def get_reg_rms_data_and_labels(subject_data):
    return get_reg_word_features_data_and_labels(subject_data, "rms")

def get_labels(subject_data, task_name):
    if task_name=="rms":
        return get_word_features_labels(subject_data, "rms")
    elif task_name=="pitch":
        return get_word_features_labels(subject_data, "pitch")
    else:
        raise RuntimeError("Not a valid data task")

def get_raw_data_and_labels(subject_data, task_name):
    if task_name=="rms":
        return get_rms_data_and_labels(subject_data)
    elif task_name=="pos":
        return get_pos_data_and_labels(subject_data)
    elif task_name=="rms_reg":
        return get_reg_rms_data_and_labels(subject_data)
    elif task_name=="pitch":
        return get_pitch_data_and_labels(subject_data)
    elif task_name=="word_onset":
        return get_word_onset_data_and_labels(subject_data)
    elif task_name=="sentence_onset":#same procedure, but subject data is different
        return get_word_onset_data_and_labels(subject_data)
    elif task_name=="pretraining":
        return get_pretraining_data_and_labels(subject_data)
    elif task_name=="spec_target_pretraining":
        return get_spec_target_pretraining_data_and_labels(subject_data)
    else:
        raise RuntimeError("Not a valid data task")

def write_outputs(subject, trial, seeg_data, labels, ordered_electrodes, extracter, output_path):

    manifest = []
    output_path = os.path.join(output_path, subject, trial)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    log.info(f'Writing embeds for subject {subject} and trial {trial}')
    for idx in tqdm(range(len(labels))):
        all_embeddings = []

        raw_neural_data = seeg_data[:,idx] 
        all_embeddings = extracter(raw_neural_data).numpy()

        save_path = os.path.join(output_path, f'{idx}.npy')
        np.save(save_path, all_embeddings)
        manifest.append(save_path)
    return manifest

def write_manifests(subject, all_manifests, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(output_path, 'manifest.tsv')
    with open(manifest_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for record in all_manifests:
            writer.writerow([record[0], record[1]])

def write_np_labels(subject, trial, seeg_data, labels, ordered_electrodes, target_extracter, output_path):
    label_path = os.path.join(output_path, 'labels.tsv')

    output_path = os.path.join(output_path, "labels", subject, trial)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    log.info(f'Writing targets for subject {subject} and trial {trial}')
    label_paths = []
    for idx in tqdm(range(len(labels))):
        all_embeddings = []
        for (e_idx, e_name) in enumerate(ordered_electrodes):
            raw_neural_data = seeg_data[e_idx,idx] 
            raw_neural_data = raw_neural_data[np.newaxis,:]
            embedding = target_extracter(raw_neural_data)
            all_embeddings.append(embedding)

        all_embeddings = np.concatenate(all_embeddings)

        save_path = os.path.join(output_path, f'{idx}.npy')
        np.save(save_path, all_embeddings)
        label_paths.append(save_path)

    with open(label_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for label in label_paths:
            writer.writerow([label])

def write_labels(output_path, labels):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    label_path = os.path.join(output_path, 'labels.tsv')
    with open(label_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for label in labels:
            writer.writerow(label)

def write_metadata(all_localization_dfs, all_ordered_electrodes, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    electrodes_path = os.path.join(output_path, 'all_ordered_electrodes.json')
    with open(electrodes_path, "w") as f:
        json.dump(all_ordered_electrodes, f)

    localization_root = os.path.join(output_path, 'localization')
    Path(localization_root).mkdir(parents=True, exist_ok=True)
    for subject,localization_df in all_localization_dfs.items():
        localization_df_path = os.path.join(localization_root, f'{subject}.csv')
        localization_df.to_csv(localization_df_path)

def get_subject_data(data_cfg_template, task_name, index_subsample=None):
    if task_name in ["rms", "pitch", "rms_reg", "pos"]:
        return SubjectData(data_cfg_template, index_subsample=index_subsample)
    elif task_name in ["pretraining", "spec_target_pretraining"]:
        return MultiElectrodeSubjectData(data_cfg_template)
    elif task_name=="word_onset":
        return WordOnsetSubjectData(data_cfg_template)
    elif task_name=="sentence_onset":
        return SentenceOnsetSubjectData(data_cfg_template)
    else:
        raise RuntimeError("Task not found")

def write_trial_data_piecemeal(subject, brain_run, extracter, data_cfg_template_copy, cfg):
    subject_data = get_subject_data(data_cfg_template_copy, cfg.data_prep.task_name) 
    transcripts = subject_data.get_transcript_dfs()
    transcript = transcripts[0]

    transcript_idxs = list(transcript.index)

    ordered_electrodes = subject_data.electrodes

    localization_df = subject_data.localization_df.set_index("Electrode", drop=True)
    localization_df = localization_df.loc[ordered_electrodes]
    #TODO make sure that seeg_data order of channel matches ordered_electrodes

    feat_idxs, labels = get_labels(subject_data, cfg.data_prep.task_name)
    #step = 1000
    #all_labels = []
    #for i in range(0, len(transcript.index), step):
    #    index_subsample = transcript_idxs[i:i+step]
    #    if len(index_subsample) == 0:
    #        break
    index_subsample = feat_idxs
    subject_data = get_subject_data(data_cfg_template_copy, cfg.data_prep.task_name, index_subsample=index_subsample) 
    seeg_data = subject_data.neural_data#TODO: check that this matches what you get from subsampling the whole data

    manifest = write_outputs(subject, brain_run, seeg_data, labels, ordered_electrodes, extracter, cfg.data_prep.output_directory)
    return manifest, labels, localization_df, ordered_electrodes

def write_trial_data(subject, brain_run, extracter, data_cfg_template_copy, cfg):
    subject_data = get_subject_data(data_cfg_template_copy, cfg.data_prep.task_name) 
    ordered_electrodes = subject_data.electrodes

    localization_df = subject_data.localization_df.set_index("Electrode", drop=True)
    localization_df = localization_df.loc[ordered_electrodes]
    
    seeg_data, labels = get_raw_data_and_labels(subject_data, cfg.data_prep.task_name)
    #TODO make sure that seeg_data order of channel matches ordered_electrodes
    manifest = write_outputs(subject, brain_run, seeg_data, labels, ordered_electrodes, extracter, cfg.data_prep.output_directory)
    return manifest, labels, localization_df, ordered_electrodes

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    #if "cached_data_array" in cfg.data:
    #    raise RuntimeError("Don't cache raw data for multi subj multi channel since it takes too much memory") 

    extracter = build_preprocessor(cfg.preprocessor)

    with open(cfg.data_prep.brain_runs, "r") as f:
        brain_runs = json.load(f)

    with open(cfg.data_prep.electrodes, "r") as f:
        electrodes = json.load(f)

    #assert electrodes.keys() == brain_runs.keys()

    all_manifests = []
    all_localization_dfs = {}
    all_ordered_electrodes = {}
    all_labels = []
    all_manifests = []

    for subject in brain_runs:
        data_cfg_template = cfg.data.copy()
        log.info(f'Writing features for {subject}')
        log.info(electrodes[subject])
        log.info(brain_runs[subject])
        data_cfg_template["subject"] = subject
        data_cfg_template["electrodes"] = electrodes[subject]

        data_cfg_template_copy = data_cfg_template.copy()
        for brain_run in brain_runs[subject]:
            log.info(f'Writing features for {brain_run}')
            data_cfg_template_copy["brain_runs"] = [brain_run]

            log.info(f'Obtaining brain data and labels {brain_run}')
            if cfg.data_prep.get("piecemeal", False):
                if cfg.data_prep.task_name not in ["rms", "pitch"]:
                    raise ValueError("Piecemeal is only intended (for now) for rms and pitch. If you want to use it for sentence or word onset, you need to change the relevant subject_data, trial_data_reader files, etc.)")
                
                manifest, labels, localization_df, ordered_electrodes = write_trial_data_piecemeal(subject, brain_run, extracter, data_cfg_template_copy, cfg)
            else:
                manifest, labels, localization_df, ordered_electrodes = write_trial_data(subject, brain_run, extracter, data_cfg_template_copy, cfg)
            log.info(f'Obtained brain data and labels {brain_run}')
            all_ordered_electrodes[subject] = ordered_electrodes
            manifest = [(p, subject) for p in manifest]
            all_manifests += manifest
            all_labels += list(labels)
            all_localization_dfs[subject] = localization_df

        write_manifests(subject, all_manifests, os.path.join(cfg.data_prep.output_directory, "subject_manifests", subject))
        write_metadata(all_localization_dfs, all_ordered_electrodes, os.path.join(cfg.data_prep.output_directory, "subject_metadata", subject))
        if cfg.data_prep.task_name in ["spec_target_pretraining"]:
            pass
        else:
            write_labels(os.path.join(cfg.data_prep.output_directory, "subject_labels", subject), labels)
    write_manifests(subject, all_manifests, cfg.data_prep.output_directory)
    write_metadata(all_localization_dfs, all_ordered_electrodes, cfg.data_prep.output_directory)
    if cfg.data_prep.task_name in ["spec_target_pretraining"]:
        target_extracter = build_preprocessor(cfg.target_preprocessor)
        write_np_labels(subject, brain_run, seeg_data, all_labels, ordered_electrodes, target_extracter, cfg.data_prep.output_directory)
    else:
        write_labels(cfg.data_prep.output_directory, all_labels)

if __name__ == "__main__":
    main()
