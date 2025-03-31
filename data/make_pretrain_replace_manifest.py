import shutil
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
import json
import numpy as np
from tqdm import tqdm as tqdm
import csv
import random
from glob import glob as glob
import pandas as pd

log = logging.getLogger(__name__)

def write_files(lines, output_path):
    with open(output_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for record in lines:
            writer.writerow(record)

def make_trial_manifest(labels_path, manifest_path, data_prep_cfg):
    if data_prep_cfg.task=="nsp_order_switch":
        return make_order_switch_task(labels_path, manifest_path)
    elif data_prep_cfg.task=="nsp_negative_any":
        return make_negative_any_task(labels_path, manifest_path)
    else:
        print("task not found")
        import pdb; pdb.set_trace()

def make_negative_any_task(labels_path, manifest_path):
    '''
    positive examples are two consecutive segments
    negative examples are two random segments
    '''
    with open(labels_path, "r") as f:
        labels = f.readlines()
        labels = [int(l.strip()) for l in labels]

    assert sorted(labels) == labels
    rows = []
    with open(manifest_path, "r") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            rows.append(row)

    manifest = []
    new_labels = []
    subject = rows[0][1]
    idx = 0
    while idx < len(rows) - 1:
        if random.random() < 0.5:
            ex1 = rows[idx][0]
            ex2 = rows[idx+1][0]
            label = 0
        else:
            ex1 = rows[idx][0]
            ex2 = rows[random.randint(0,len(rows)-1)][0]
            label = 1
        ex3 = rows[random.randint(0,len(rows)-1)][0]
        manifest.append([ex1, ex2, ex3, subject])
        new_labels.append([label])
        idx += 1
    return manifest, new_labels


def make_order_switch_task(labels_path, manifest_path):
    with open(labels_path, "r") as f:
        labels = f.readlines()
        labels = [int(l.strip()) for l in labels]

    assert sorted(labels) == labels
    rows = []
    with open(manifest_path, "r") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            rows.append(row)

    manifest = []
    new_labels = []
    subject = rows[0][1]
    idx = 0
    while idx < len(rows) - 1:
        if random.random() < 0.5:
            ex1 = rows[idx][0]
            ex2 = rows[idx+1][0]
            label = 0
        else:
            ex1 = rows[idx+1][0]
            ex2 = rows[idx][0]
            label = 1
        manifest.append([ex1, ex2, subject])
        new_labels.append([label])
        idx += 2
    return manifest, new_labels

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    localization_src = os.path.join(cfg.data_prep.source_dir, "localization")
    assert os.path.exists(localization_src)

    localization_dst = os.path.join(cfg.data_prep.output_dir, "localization")
    if not os.path.exists(localization_dst):
        shutil.copytree(localization_src, localization_dst)

    all_electrode_jsons = glob(os.path.join(cfg.data_prep.source_dir, "ordered_electrodes", "*"))

    all_electrode_jsons = glob(os.path.join(cfg.data_prep.source_dir, "ordered_electrodes", "*"))
    all_electrodes = {}
    for electrode_json in all_electrode_jsons:
        subject = Path(electrode_json).stem
        with open(electrode_json, "r") as f:
            all_electrodes[subject] = json.load(f)

    Path(cfg.data_prep.output_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(cfg.data_prep.output_dir, "all_ordered_electrodes.json"), "w") as f:
        json.dump(all_electrodes, f)

    all_manifests, all_labels = [], []
    for subject in all_electrodes.keys():
        subject_path = os.path.join(cfg.data_prep.source_dir, subject)
        for trial_path in glob(os.path.join(subject_path, "*")):
            labels_path = os.path.join(trial_path, "labels.tsv")
            manifest_path = os.path.join(trial_path, "manifest.tsv")
            manifest, labels = make_trial_manifest(labels_path, manifest_path, cfg.data_prep)
            all_manifests += manifest
            all_labels += labels

    manifest_path = os.path.join(cfg.data_prep.output_dir, 'manifest.tsv')
    write_files(all_manifests, manifest_path)

    labels_path = os.path.join(cfg.data_prep.output_dir, 'labels.tsv')
    write_files(all_labels, labels_path)


if __name__ == "__main__":
    main()


