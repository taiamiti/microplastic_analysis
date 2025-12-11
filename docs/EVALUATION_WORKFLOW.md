# Evaluation Workflow Documentation

This document describes the new modular evaluation system for both semantic and instance segmentation.

## Overview

The new `src/modeling/evaluate_segmentation.py` script provides a CLI for:

1. **Adding predictions** to existing FiftyOne datasets with unique field naming
2. **Evaluating semantic segmentation** (IoU, Dice, precision, recall)
3. **Converting to instance segmentation** (both GT and predictions)
4. **Evaluating instance segmentation** (mAP, precision, recall per instance)

## Field Naming Convention

To support multiple model experiments on the same dataset, fields are named using the model config:

- **Semantic predictions**: `predictions_{config_name}`
- **Instance ground truth**: `inst_ground_truth`
- **Instance predictions**: `inst_predictions_{config_name}`
- **Evaluation keys**: `eval_{config_name}` or `eval_inst_{config_name}`

### Config Name Extraction

Long config names are automatically shortened for readability:

```python
# Original config
"fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py"

# Extracted name (used in fields)
"400x400_train_test"
```

You can test the extraction:
```bash
python src/modeling/evaluate_segmentation.py extract_config_name \
  "fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py"
```

## CLI Commands

### 1. Add Predictions to Dataset

Add prediction masks from an inference directory to an existing FiftyOne dataset:

```bash
python src/modeling/evaluate_segmentation.py add_predictions \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/fcn-unet_400x400/inference \
  --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py
```

**Parameters:**
- `dataset_name`: Existing FiftyOne dataset name
- `predictions_dir`: Root directory with predictions organized by lot
- `config_name`: Config filename (with or without .py extension)
- `pred_field`: (Optional) Custom field name instead of auto-generated
- `filter_tags`: (Optional) Only add predictions to samples with this tag (e.g., "test")
- `label_ext`: (Optional) Mask file extension (default: ".png")

**Expected directory structure:**
```
predictions_dir/
├── lot1-20-04-2023-benitiers/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── lot2-30-05-2023-tak_nacl/
│   ├── image1.png
│   └── ...
```

### 2. Evaluate Semantic Segmentation

Evaluate semantic segmentation metrics (IoU, Dice, precision, recall):

```bash
python src/modeling/evaluate_segmentation.py eval_semantic \
  --dataset_name mp_dataset \
  --pred_field predictions_400x400_train_test \
  --gt_field ground_truth \
  --filter_tags test
```

**Parameters:**
- `dataset_name`: FiftyOne dataset name
- `pred_field`: Field with predictions (e.g., "predictions_400x400_train_test")
- `gt_field`: Field with ground truth (default: "ground_truth")
- `eval_key`: (Optional) Custom evaluation key (default: "eval_{config_name}")
- `filter_tags`: (Optional) Filter to samples with this tag (default: "test")
- `mask_targets`: (Optional) Dict mapping pixel values to class names (default: {0: "background", 255: "mp"})

**Output:**
- Per-sample metrics: accuracy, precision, recall
- Classification report with per-class IoU, Dice, etc.
- Metrics saved to dataset under `eval_key`

### 3. Convert to Instance Segmentation

Convert semantic segmentation masks to instance segmentation detections:

```bash
# Convert ground truth
python src/modeling/evaluate_segmentation.py convert_to_instances \
  --dataset_name mp_dataset \
  --mask_field ground_truth \
  --det_field inst_ground_truth \
  --compute_scores True

# Convert predictions
python src/modeling/evaluate_segmentation.py convert_to_instances \
  --dataset_name mp_dataset \
  --mask_field predictions_400x400_train_test \
  --det_field inst_predictions_400x400_train_test \
  --compute_scores True
```

**Parameters:**
- `dataset_name`: FiftyOne dataset name
- `mask_field`: Field with semantic masks to convert
- `det_field`: (Optional) Output detection field (default: "inst_{mask_field}")
- `compute_scores`: Whether to compute contrast scores and RGB values (default: True)
- `min_area`: Minimum detection area in pixels (default: 40)
- `max_area`: Maximum detection area in pixels (default: 160000)
- `filter_tags`: (Optional) Only process samples with this tag
- `batch_size`: Save frequency (default: 100 samples)

**Detection Attributes:**
Each instance detection includes MP-VAT shape descriptors:
- `area`, `perimeter`
- `feret_diameter`, `feret_degree`
- `circularity`, `roundness`
- `mp_shape`: Classification (Fibers, Fragments, Particles)
- `score`: Contrast-based quality metric (if `compute_scores=True`)
- `mean_red`, `mean_green`, `mean_blue`: RGB values (if `compute_scores=True`)

### 4. Evaluate Instance Segmentation

Evaluate instance segmentation metrics (mAP, precision, recall):

```bash
python src/modeling/evaluate_segmentation.py eval_instances \
  --dataset_name mp_dataset \
  --pred_field inst_predictions_400x400_train_test \
  --gt_field inst_ground_truth \
  --filter_tags test
```

**Parameters:**
- `dataset_name`: FiftyOne dataset name
- `pred_field`: Field with predicted detections
- `gt_field`: Field with ground truth detections (default: "inst_ground_truth")
- `eval_key`: (Optional) Custom evaluation key
- `filter_tags`: (Optional) Filter to samples with this tag (default: "test")
- `iou_thresholds`: (Optional) List of IoU thresholds (default: [0.5, 0.75, 0.9])
- `compute_mAP`: Whether to compute mAP metrics (default: True)

**Output:**
- mAP scores at different IoU thresholds
- Per-class precision, recall, F1-score
- Confusion matrix

### 5. Full Evaluation Pipeline

Run the complete workflow in one command:

```bash
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/fcn-unet_400x400/inference \
  --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py \
  --filter_tags test
```

**Parameters:**
- All parameters from previous commands
- `skip_add_predictions`: Skip adding predictions (if already added)
- `skip_semantic`: Skip semantic evaluation
- `skip_instance`: Skip instance evaluation and conversion

**Workflow steps:**
1. Add predictions to dataset
2. Evaluate semantic segmentation
3. Convert ground truth to instances (if not exists)
4. Convert predictions to instances
5. Evaluate instance segmentation

## Complete Examples

### Example 1: Evaluate a New Model

```bash
# Set environment
export PYTHONPATH=$PWD

# Full pipeline for a 400x400 model trained on train_test protocol
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/inference \
  --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py \
  --filter_tags test \
  --compute_scores True \
  --min_area 40 \
  --max_area 160000
```

### Example 2: Compare Multiple Models

```bash
# Model 1: 400x400 train_test
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model1/inference \
  --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py

# Model 2: 256x256 sed_intra_inter_ile
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model2/inference \
  --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_sed_intra_inter_ile.py

# Now mp_dataset contains:
# - predictions_400x400_train_test
# - predictions_256x256_sed_intra_inter_ile
# - inst_predictions_400x400_train_test
# - inst_predictions_256x256_sed_intra_inter_ile
# Each with their own evaluation metrics
```

### Example 3: Step-by-Step Evaluation

```bash
# Step 1: Add predictions
python src/modeling/evaluate_segmentation.py add_predictions \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/my_model/inference \
  --config_name my_model_config.py

# Step 2: Evaluate semantic
python src/modeling/evaluate_segmentation.py eval_semantic \
  --dataset_name mp_dataset \
  --pred_field predictions_my_model_config

# Step 3: Convert GT to instances (once)
python src/modeling/evaluate_segmentation.py convert_to_instances \
  --dataset_name mp_dataset \
  --mask_field ground_truth

# Step 4: Convert predictions to instances
python src/modeling/evaluate_segmentation.py convert_to_instances \
  --dataset_name mp_dataset \
  --mask_field predictions_my_model_config

# Step 5: Evaluate instances
python src/modeling/evaluate_segmentation.py eval_instances \
  --dataset_name mp_dataset \
  --pred_field inst_predictions_my_model_config
```

## Visualizing Results in FiftyOne

After running evaluations, visualize results in the FiftyOne App:

```python
import fiftyone as fo

# Load dataset
dataset = fo.load_dataset("mp_dataset")

# View test samples with semantic predictions
view = dataset.match_tags("test").exists("predictions_400x400_train_test")

# Launch FiftyOne App
session = fo.launch_app(view)
```

In the FiftyOne App, you can:
- Toggle between different prediction fields
- Compare ground truth vs predictions side-by-side
- Filter by evaluation metrics (accuracy, precision, recall)
- Visualize instance detections with bounding boxes
- Sort by detection scores or shape descriptors

## Troubleshooting

### Issue: Predictions not found

**Error:** `Warning: X samples missing prediction masks`

**Solution:** Verify predictions directory structure matches expected format:
```bash
ls data/modeling/work_dirs/model/inference/
# Should show: lot1-20-04-2023-benitiers/ lot2-... etc.

ls data/modeling/work_dirs/model/inference/lot1-20-04-2023-benitiers/
# Should show: image1.png image2.png etc.
```

### Issue: No samples found for evaluation

**Error:** `No samples found with both prediction and ground truth fields!`

**Solution:** Check that:
1. Predictions were added successfully (check dataset fields)
2. The `filter_tags` parameter matches your dataset tags
3. Both GT and prediction fields exist on the same samples

```python
import fiftyone as fo
dataset = fo.load_dataset("mp_dataset")

# Check fields
print(dataset.get_field_schema())

# Check tags
print(dataset.distinct("tags"))

# Check sample counts
print(f"Total: {len(dataset)}")
print(f"With GT: {len(dataset.exists('ground_truth'))}")
print(f"With predictions: {len(dataset.exists('predictions_400x400_train_test'))}")
```

### Issue: Config name too long

**Error:** Field names become unwieldy

**Solution:** Use the `pred_field` parameter to specify a custom short name:

```bash
python src/modeling/evaluate_segmentation.py add_predictions \
  --dataset_name mp_dataset \
  --predictions_dir path/to/inference \
  --config_name very_long_config_name.py \
  --pred_field predictions_mymodel
```

## Migration from Old Workflow

If you're using the old `run_fiftyone_eval.py`:

**Old workflow:**
```bash
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  data/modeling/work_dirs/model/inference \
  --eval_bool True
```

**New workflow:**
```bash
# First time: Create persistent dataset (if not exists)
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  data/modeling/work_dirs/model/inference \
  --eval_bool False

# Then use new evaluation script
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name fo_eval_dataset \
  --predictions_dir data/modeling/work_dirs/model/inference \
  --config_name your_config.py
```

**Benefits of new workflow:**
- ✅ Persistent dataset (no re-creation needed)
- ✅ Multiple models on same dataset
- ✅ Both semantic and instance evaluation
- ✅ Unique field names prevent conflicts
- ✅ Modular steps for debugging
