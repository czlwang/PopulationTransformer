REPO_DIR="/path/to/PopulationTransformer"

SUBJECT=sub_1; TASK=rms; python3 -m data.make_subject_specific_manifest \
+data_prep=subject_specific_manifest \
++data_prep.data_path=${REPO_DIR}/saved_examples/all_test_${TASK} \
++data_prep.subj=${SUBJECT} \
++data_prep.out_path=${REPO_DIR}/saved_examples/${SUBJECT}_${TASK}_cr