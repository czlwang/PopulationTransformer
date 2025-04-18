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