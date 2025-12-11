#!/bin/bash

export PYTHONPATH=$PWD

# Example for article beni evaluations
config=fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test
dataset=mp_article_beni

# Step 1: Add predictions to dataset
python src/modeling/add_predictions.py \
  --dataset_name $dataset \
  --predictions_dir work_dirs/$config/inference \
  --config_name $config.py \
  --filter_tags test

# Step 2: Evaluate semantic segmentation (IoU, Dice, precision, recall)
python src/modeling/evaluate_sem_seg.py \
  --dataset_name $dataset \
  --pred_field predictions_400x400_train_test \
  --filter_tags test

## Step 3: Convert ground truth to instances (ONCE per dataset)
#python src/modeling/convert_to_instance_segmentation.py \
#  --dataset_name $dataset \
#  --mask_field ground_truth \
#  --det_field inst_ground_truth \
#  --filter_tags test

# Step 4: Convert predictions to instances
python src/modeling/convert_to_instance_segmentation.py \
  --dataset_name $dataset \
  --mask_field predictions_400x400_train_test \
  --det_field inst_predictions_400x400_train_test \
  --filter_tags test

# Step 5: Evaluate instance segmentation (mAP, per-instance metrics)
python src/modeling/evaluate_inst_seg.py \
  --dataset_name $dataset \
  --pred_field inst_predictions_400x400_train_test \
  --gt_field inst_ground_truth \
  --filter_tags test