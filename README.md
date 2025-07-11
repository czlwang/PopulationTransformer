# PopulationTransformer
This is the code for the [PopulationTransformer](https://arxiv.org/abs/2406.03044). 

[![Paper](https://img.shields.io/badge/arXiv-2406.03044-red)](https://arxiv.org/abs/2406.03044)
[![Website](https://img.shields.io/badge/Website-blue)](https://glchau.github.io/population-transformer/)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/PopulationTransformer/popt_brainbert_stft)
[![Dataset](https://img.shields.io/badge/Dataset-teal)](https://braintreebank.dev/)

<p align="left">
    &nbsp;<a href="#quick-start">Quick Start</a>
    | &nbsp;<a href="#prerequisites">Prerequisites</a>
    | &nbsp;<a href="#data">Data</a>
    | &nbsp;<a href="#pretraining">Pretraining</a>
    | &nbsp;<a href="#fine-tuning">Fine-tuning</a>
    | &nbsp;<a href="#citation">Citation</a>
</p>

## Quick Start
With the [requirements](#prerequisites) and [data](#data) installed, run these scripts for their desired outcomes: 

| Script    | Outcome |
| -------- | ------- |
| `0_write_pretraining_data.sh`  | Write pretraining data |
| `1_create_pretraining_manifest.sh` | Create the pretraining manifest file |
| `2_run_pretraining.sh`    | Run pretraining |
| `3_write_finetuning_data.sh`    | Write fine-tuning data |
| `4_create_finetuning_manifest.sh`    | Create fine-tuning manifest file |
| `5_run_finetuning.sh`    | Run fine-tuning |

More details about each script in [pretraining](#pretraining) and [fine-tuning](#fine-tuning) sections below. 

## Prerequisites
Requirements:
- pytorch >= 1.13.1
```bash
pip install -r requirements.txt
```

## Data
If using the data from the Brain Treebank, the data can be downloaded from [braintreebank.dev](https://braintreebank.dev).

The following commands expects the Brain Treebank data to have the following structure:
```
/braintreebank_data
  |_electrode_labels
  |_subject_metadata
  |_localization
  |_all_subject_data
    |_sub_*_trial*.h5
```

## Pretraining
<details>
  <summary>Details</summary>
  
The below commands assume that you will be using the PopulationTransformer in conjunction with BrainBERT. 
If you are, you will need to download the [BrainBERT](https://github.com/czlwang/BrainBERT) weights.

First, we write the BrainBERT features for pre-training. This command takes a list of brain recordings (see below) and creates a training dataset from their BrainBERT representations:
```bash
REPO_DIR="/path/to/PopulationTransformer"
BRAINTREEBANK_DIR="/path/to/braintreebank_data"

python3 -m data.write_nsp_pretraining_data \
+preprocessor=multi_elec_spec_pretrained \
++preprocessor.upstream_ckpt=${REPO_DIR}/pretrained_weights/stft_large_pretrained.pth \
+data_prep=pretrain_multi_subj_multi_chan_template \
++data_prep.task_name=nsp_pretraining \
++data_prep.brain_runs=${REPO_DIR}/trial_selections/pretrain_split_trials.json \
++data_prep.electrodes=${REPO_DIR}/electrode_selections/clean_laplacian.json \
++data_prep.output_directory=${REPO_DIR}/saved_examples/cr_pretrain_examples \
+data=pretraining_subject_data_template \
++data.cached_transcript_aligns=${REPO_DIR}/semantics/saved_aligns \
++data.cached_data_array=${REPO_DIR}/cached_data_arrays/ \
++data.raw_brain_data_dir=${BRAINTREEBANK_DIR}  
```
Salient arguments:
- Input:
    - `preprocessor.upstream_ckpt` is the path to the BrainBERT weights
    - `preprocessor.brain_runs` is the path to a json file of the following format: `{<sub_name>: [trial_name]}`. This specifies the brain recording files that will be used.
    - `data_prep.electrodes` is the path to a json file of the following format `{<sub_name>: [electrode_name]}`. Similar to the above, this specifies which channels will be used.
    - `data.raw_brain_data_dir` is the path to the root of the Brain Treebank data (see the Data section above)
- Output:
    - `data_prep.output_directory` is the path where the output will be written
    - `data.cached_data_array` is the path to an (optional) cache where intermediate outputs can be written for faster processing 

Next, we need to create a manifest for all the training examples we've just created. 
```bash
REPO_DIR="/path/to/PopulationTransformer"

python3 -m data.make_pretrain_replace_manifest +data_prep=combine_nsp_datasets \
++data_prep.source_dir=${REPO_DIR}/saved_examples/cr_pretrain_examples \
++data_prep.output_dir=${REPO_DIR}/saved_examples/nsp_replace_task-0_5s \
++data_prep.task="nsp_negative_any"
```
Salient arguments:
- Input:
    - `data_prep.source_dir` should match `data_prep.output_dir` from above.
- Output:
    - `data_prep.output_dir` is the path where the output will be written.

Now, we can run the pretraining
```bash
REPO_DIR="/path/to/PopulationTransformer"

python3 run_train.py \
+exp=multi_elec_pretrain \
++exp.runner.device=cuda \
+data=nsp_replace_only_pretrain \
++data.data_path=${REPO_DIR}/saved_examples/nsp_replace_task-0_5s \
++data.saved_data_split=${REPO_DIR}/saved_data_splits/pretrain_split \
++data.test_data_cfg.name=nsp_replace_only_deterministic \
++data.test_data_cfg.data_path=${REPO_DIR}/saved_examples/nsp_replace_task-0_5s \
+model=pt_custom_model \
+task=nsp_replace_only_pretrain \
+criterion=nsp_replace_only_pretrain \
+preprocessor=empty_preprocessor
```
Salient arguments:
- Input:
    - `data.data_path` should match the `data_prep.output_dir` from the manifest creation step above.
- Output: 
    - The final weights will be saved in an automatically created directory under `outputs`.
</details>


## Fine-tuning
<details>
  <summary>Details</summary>
  
Now, let's write the BrainBERT features for a finetuning task. For this example, let's decode volume (rms) from one electrode over the course of one trial.
```bash
REPO_DIR="/path/to/PopulationTransformer"
BRAINTREEBANK_DIR="/path/to/braintreebank_data"

python3 -m data.write_multi_subject_multi_channel \
+data_prep=pretrain_multi_subj_multi_chan_template \
++data_prep.task_name=rms \
++data_prep.brain_runs=${REPO_DIR}/trial_selections/test_trials.json \
++data_prep.electrodes=${REPO_DIR}/electrode_selections/clean_laplacian.json \
++data_prep.output_directory=${REPO_DIR}/saved_examples/all_test_rms \
+preprocessor=multi_elec_spec_pretrained \
++preprocessor.upstream_ckpt=${REPO_DIR}/pretrained_weights/stft_large_pretrained.pth \
+data=subject_data_template \
++data.cached_transcript_aligns=${REPO_DIR}/semantics/saved_aligns \
++data.cached_data_array=${REPO_DIR}/cached_data_arrays/ \
++data.raw_brain_data_dir=${BRAINTREEBANK_DIR}/ \
++data.movie_transcripts_dir=${BRAINTREEBANK_DIR}/transcripts
```
- Inputs:
    - `data_prep.electrodes` and `data_prep.brain_runs` as in Pretraining, these files specify the trials and channels that will be used to create the dataset.
- Outputs:
    - `data_prep.output_directory` is the path to where the BrainBERT embeddings will be written.


Let's write the manifest for this decoding task.
```bash
REPO_DIR="/path/to/PopulationTransformer"

SUBJECT=sub_1; TASK=rms; python3 -m data.make_subject_specific_manifest \
+data_prep=subject_specific_manifest \
++data_prep.data_path=${REPO_DIR}/saved_examples/all_test_${TASK} \
++data_prep.subj=${SUBJECT} \
++data_prep.out_path=${REPO_DIR}/saved_examples/${SUBJECT}_${TASK}_cr
```
- Inputs:
    - `data_prep.data_path` should match the `output_directory` given above

Now, we are ready to run the finetuning. You an either fine-tune a model that you have pre-trained yourself, or use a model from [our huggingface repo](https://huggingface.co/PopulationTransformer).
```bash
REPO_DIR="/path/to/PopulationTransformer"


SUBJECT=sub_1; TASK=rms; N=1; NAME=popt_brainbert_stft; WEIGHTS=pretrained_popt_brainbert_stft; 
python3 run_train.py \
+exp=multi_elec_feature_extract \
++exp.runner.results_dir=${REPO_DIR}/outputs/${SUBJECT}_${TASK}_top${N}_${NAME} \
++exp.runner.save_checkpoints=False \
++model.frozen_upstream=False \
+task=pt_feature_extract_coords \
+criterion=pt_feature_extract_coords_criterion \
+preprocessor=empty_preprocessor \
+data=pt_supervised_task_coords \
++data.data_path=${REPO_DIR}/saved_examples/${SUBJECT}_${TASK}_cr \
++data.saved_data_split=${REPO_DIR}/saved_data_splits/${SUBJECT}_${TASK}_fine_tuning \
++data.sub_sample_electrodes=${REPO_DIR}/electrode_selections/debug_electrodes.json \
+model=pt_downstream_model \
++model.upstream_path=${REPO_DIR}/pretrained_weights/${WEIGHTS}.pth 
```
- Inputs:
    - `data.data_path` should match the `out_path` of the manifest creation step above.
    - `model.upstream_path` should be a path to the weights from pretraining --- either from the steps above, or from [the huggingface repo](https://huggingface.co/PopulationTransformer).
    - `data.saved_data_split` is a path to where the indices for train/val/test splits will be written. You can use this to ensure that splits are consistent between runs.
- Outputs:
    - `exp.runner.results_dir` will contain performance metrics (f1, ROC-AUC) on the test set.
</details>

## Citation
```bibtex
@inproceedings{
      chau_wang_2025_population,
      title={Population Transformer: Learning Population-level Representations of Neural Activity},
      url={https://openreview.net/forum?id=FVuqJt3c4L},
      booktitle={The Thirteenth International Conference on Learning Representations},
      author={Chau, Geeling and Wang, Christopher and Talukder, Sabera J and Subramaniam, Vighnesh and Soedarmadji, Saraswati and Yue, Yisong and Katz, Boris and Barbu, Andrei},
      year={2025}
}
```
