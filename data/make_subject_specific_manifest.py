from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
import csv
import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import shutil

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Making subject specific manifest and metadata files")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')
    
    data_path = cfg.data_prep.data_path

    manifest_path = os.path.join(data_path, "manifest.tsv")
    assert os.path.exists(manifest_path)
    manifest = []
    with open(manifest_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            manifest.append(row)

    label_path = os.path.join(data_path, "labels.tsv")
    assert os.path.exists(label_path)
    labels = []
    with open(label_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            labels.append(row)

    out_path = cfg.data_prep.out_path
    Path(out_path).mkdir(exist_ok=True, parents=True)
    src = os.path.join(data_path, "localization")
    dest = os.path.join(out_path, "localization")
    if not os.path.exists(dest):
        shutil.copytree(src, dest)

    src = os.path.join(data_path, "all_ordered_electrodes.json")
    dest = os.path.join(out_path, "all_ordered_electrodes.json")
    if not os.path.exists(dest):
        shutil.copy(src, dest)

    new_manifest, new_labels = [], []
    for manifest_record, labels_record in zip(manifest, labels):
        if manifest_record[1] == cfg.data_prep.subj:
            new_manifest.append(manifest_record)
            new_labels.append(labels_record)

    manifest_path = os.path.join(out_path, "manifest.tsv")
    with open(manifest_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for record in new_manifest:
            writer.writerow(record)
   
    labels_path = os.path.join(out_path, "labels.tsv")
    with open(labels_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for record in new_labels:
            writer.writerow(record)
  
if __name__ == "__main__":
    main()


