export PYTHONPATH=$PWD

# define var
lot=lot1-20-04-2023-benitiers

conda activate map2
# ingest data
python src/pipeline.py ingest_data_subset configs/test_config_lot1.yaml $lot
# create composite
python src/pipeline.py create_composite_subset configs/test_config_lot1.yaml $lot

# create task
python src/pipeline.py create_tasks configs/test_config_lot1.yaml $lot

# copy labkit generated annotations to annotated_data (manual op)
# copy matching names
python src/pipeline.py matching_old_names_with_new configs/test_config_lot1.yaml $lot

# generate annotated subset
python src/pipeline.py generate_annotated_subset configs/test_config_lot1.yaml lot1-20-04-2023-benitiers

# convert annotated dataset (load all datasets then convert)
python src/pipeline.py prepare_dataset_for_openmmseg configs/test_config_lot1.yaml

# train
conda activate openmmlab
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test.py \
--work-dir data/test_config_lot1/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test

# inference (for test we simply copy the checkpoint to the correct path and try it on labelled data
# so that we can use fiftyone for evaluations)
python mmsegmentation/tools/inference.py \
--model_cfg mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test.py \
--model_ckpts data/test_config_lot1/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/best_mIoU_iter_6200.pth \
--img_folder data/test_config_lot1/generate_annotated_dataset/lot1-20-04-2023-benitiers/data \
--save_folder data/test_config_lot1/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference/lot1-20-04-2023-benitiers

# evaluate using fiftyone
python src/modeling/run_fiftyone_eval.py \
data/test_config_lot1/generate_annotated_dataset \
data/test_config_lot1/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference/lot1-20-04-2023-benitiers \
--eval_bool True

