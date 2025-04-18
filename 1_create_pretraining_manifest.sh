REPO_DIR="/path/to/PopulationTransformer"

python3 -m data.make_pretrain_replace_manifest +data_prep=combine_nsp_datasets \
++data_prep.source_dir=${REPO_DIR}/saved_examples/cr_pretrain_examples \
++data_prep.output_dir=${REPO_DIR}/saved_examples/nsp_replace_task-0_5s \
++data_prep.task="nsp_negative_any"