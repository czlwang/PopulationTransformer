from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
from datasets import build_dataset
from preprocessors import build_preprocessor
from data.subject_data import SubjectData
from data.multi_electrode_subj_data import MultiElectrodeSubjectData
from data.speech_nonspeech_subject_data import WordOnsetSubjectData
import json
import numpy as np
from tqdm import tqdm as tqdm
import csv
from scipy.stats import zscore
import random

log = logging.getLogger(__name__)

def get_raw_data_and_labels(subject_data, task_name, cfg, separation_interval=0.5):#separation_interval is 0.5s
    if task_name=="nsp_pretraining":
        return get_pretraining_data_and_labels(subject_data, separation_interval, cfg)
    else:
        raise RuntimeError("Not a valid data task")

def get_pretraining_data_and_labels(subject_data, separation_interval, cfg):
    seeg_data = subject_data.neural_data

    #reshape the data into one continuous time slice and then chunk it up
    interval_length = seeg_data.shape[-1]
    seeg_data = seeg_data.reshape(seeg_data.shape[0], seeg_data.shape[1]*seeg_data.shape[2])

    interval_step = int(separation_interval*2048)#hardcode sample rate

    max_n_examples = float('inf')
    if 'max_n_examples' in cfg.data_prep:
       max_n_examples = int(cfg.data_prep.max_n_examples )

    labels, seeg_exs = [], []
    idx = 0
    while idx < seeg_data.shape[1]-interval_length and len(labels) < max_n_examples:
        ex = seeg_data[:,idx:idx+interval_length]
        seeg_exs.append(ex)
        labels.append(idx)
        idx += interval_step
    return seeg_exs, labels

def write_outputs(subject, trial, seeg_exs, labels, ordered_electrodes, extracter, output_path):
    manifest = []
    output_path = os.path.join(output_path, subject, trial)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    log.info(f'Writing embeds for subject {subject} and trial {trial}')
    for idx in tqdm(range(len(labels))):
        raw_neural_data = seeg_exs[idx]
        all_embeddings = extracter(raw_neural_data).numpy()

        save_path = os.path.join(output_path, f'{idx}.npy')
        np.save(save_path, all_embeddings)
        manifest.append(save_path)
    return manifest

def write_manifests(all_manifests, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(output_path, 'manifest.tsv')
    with open(manifest_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for record in all_manifests:
            writer.writerow([record[0], record[1]])

def write_labels(output_path, labels):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    label_path = os.path.join(output_path, 'labels.tsv')
    with open(label_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for label in labels:
            writer.writerow([label])

def write_metadata(subject, localization_df, ordered_electrodes, output_path):
    ordered_electrodes_root = os.path.join(output_path, 'ordered_electrodes')
    Path(ordered_electrodes_root).mkdir(parents=True, exist_ok=True)
    electrodes_path = os.path.join(ordered_electrodes_root, f'{subject}.json')
    with open(electrodes_path, "w") as f:
        json.dump(ordered_electrodes, f)

    localization_root = os.path.join(output_path, 'localization')
    Path(localization_root).mkdir(parents=True, exist_ok=True)
    localization_df_path = os.path.join(localization_root, f'{subject}.csv')
    localization_df.to_csv(localization_df_path)

def get_subject_data(data_cfg_template, task_name):
    if task_name in ["nsp_pretraining"]:
        return MultiElectrodeSubjectData(data_cfg_template)
    else:
        raise RuntimeError("Task not found")

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

    all_ordered_electrodes = {}

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

            subject_data = get_subject_data(data_cfg_template_copy, cfg.data_prep.task_name) 
            ordered_electrodes = subject_data.electrodes

            localization_df = subject_data.localization_df.set_index("Electrode", drop=True)
            localization_df = localization_df.loc[ordered_electrodes]
            
            log.info(f'Obtaining brain data and labels {brain_run}')
            seeg_exs, labels = get_raw_data_and_labels(subject_data, cfg.data_prep.task_name, cfg, separation_interval=cfg.data_prep.separation_interval)
            log.info(f'Obtained brain data and labels {brain_run}')
            manifest = write_outputs(subject, brain_run, seeg_exs, labels, ordered_electrodes, extracter, cfg.data_prep.output_directory)
            manifest = [(p, subject) for p in manifest]
            write_manifests(manifest, os.path.join(cfg.data_prep.output_directory, subject, brain_run))
            write_labels(os.path.join(cfg.data_prep.output_directory, subject, brain_run), list(labels))
        write_metadata(subject, localization_df, ordered_electrodes, cfg.data_prep.output_directory)

if __name__ == "__main__":
    main()
