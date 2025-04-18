REPO_DIR="/path/to/PopulationTransformer"


SUBJECT=sub_1; TASK=rms; N=1; NAME=popt_brainbert_stft; WEIGHTS=popt_brainbert_stft; 
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
++model.upstream_path=${REPO_DIR}/pretrained_weights/${WEIGHTS}.pth \