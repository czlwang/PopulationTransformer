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