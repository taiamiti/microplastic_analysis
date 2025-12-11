# Migration Guide: New Evaluation System

## What Changed?

We've introduced a new modular evaluation system (`src/modeling/evaluate_segmentation.py`) that replaces the old monolithic workflow while maintaining backward compatibility.

## Old Workflow vs New Workflow

### Old Workflow (Still Supported)

```bash
# Create dataset + evaluate in one step
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  data/modeling/work_dirs/model/inference \
  --eval_bool True
```

**Limitations:**
- ❌ Creates dataset from scratch each time
- ❌ Only semantic segmentation evaluation
- ❌ No instance segmentation metrics
- ❌ Can't compare multiple models easily
- ❌ Overwrites previous predictions

### New Workflow (Recommended)

```bash
# All-in-one command
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model/inference \
  --config_name your_config.py
```

**Benefits:**
- ✅ Persistent dataset (no re-creation)
- ✅ Both semantic and instance evaluation
- ✅ Multiple models on same dataset
- ✅ Config-based field naming (no conflicts)
- ✅ Modular steps for debugging

## Migration Steps

### Step 1: Create Persistent Dataset (One Time)

If you don't already have a persistent FiftyOne dataset:

```bash
# Using old script to create the dataset
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  data/modeling/work_dirs/placeholder/inference \
  --dataset_name mp_dataset \
  --eval_bool False

# Or create from Python
import fiftyone as fo

dataset = fo.Dataset(name="mp_dataset", persistent=True)
dataset.add_dir(
    dataset_dir="data/processed/generate_annotated_dataset",
    dataset_type=fo.types.FiftyOneDataset
)
```

### Step 2: Use New Evaluation Script

```bash
# Evaluate your model
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/your_model/inference \
  --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py
```

### Step 3: Compare Multiple Models

```bash
# Evaluate Model 1
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model1/inference \
  --config_name config1.py

# Evaluate Model 2
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model2/inference \
  --config_name config2.py

# Now you can compare both models in FiftyOne App
```

## Field Naming Convention

### Old System
- Always used the same field name: `prediction`
- Overwritten on each evaluation

### New System
- Config-based naming: `predictions_{config_name}`
- Example: `predictions_400x400_train_test`
- No conflicts between models

## Instance Segmentation

### Old System
- Manual conversion required using separate script
- No automatic evaluation metrics

### New System
- Automatic conversion during `run_full_eval`
- Instance fields: `inst_ground_truth`, `inst_predictions_{config_name}`
- Full mAP metrics computed automatically

## Backward Compatibility

The old `run_fiftyone_eval.py` script is **still available** and **fully supported**. You can continue using it if:
- You prefer the old workflow
- You only need semantic segmentation
- You don't need to compare multiple models

## Example: Complete Migration

### Before (Old Workflow)

```bash
# Model 1
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  data/modeling/work_dirs/model1/inference \
  --eval_bool True

# To evaluate Model 2, need to:
# 1. Close FiftyOne App
# 2. Re-run script (overwrites Model 1 results)
# 3. Can't compare side-by-side
```

### After (New Workflow)

```bash
# Create dataset once
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  placeholder \
  --dataset_name mp_dataset \
  --eval_bool False

# Evaluate Model 1
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model1/inference \
  --config_name model1_config.py

# Evaluate Model 2 (adds to same dataset)
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model2/inference \
  --config_name model2_config.py

# Compare in FiftyOne App
import fiftyone as fo
dataset = fo.load_dataset("mp_dataset")
session = fo.launch_app(dataset.match_tags("test"))
# Toggle between predictions_model1_config and predictions_model2_config
```

## Troubleshooting

### Issue: Dataset already exists

```python
# Delete old dataset if needed
import fiftyone as fo
fo.delete_dataset("mp_dataset")
```

### Issue: Predictions field name too long

```bash
# Use custom field name
python src/modeling/evaluate_segmentation.py add_predictions \
  --dataset_name mp_dataset \
  --predictions_dir path/to/inference \
  --config_name very_long_config_name.py \
  --pred_field predictions_short_name
```

### Issue: Want to re-run evaluation

```bash
# Skip adding predictions if already added
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir path/to/inference \
  --config_name your_config.py \
  --skip_add_predictions True
```

## Getting Help

- Full documentation: [docs/EVALUATION_WORKFLOW.md](EVALUATION_WORKFLOW.md)
- Examples: See README.md [Step 8.3](#step-83--evaluate-models-semantic-and-instance-segmentation)
- Testing: Run `python test_evaluate_segmentation.py`

## Summary

| Feature | Old System | New System |
|---------|-----------|------------|
| Semantic Evaluation | ✅ | ✅ |
| Instance Evaluation | ❌ | ✅ |
| Persistent Dataset | ❌ | ✅ |
| Multiple Models | ❌ | ✅ |
| Field Naming | Fixed | Config-based |
| Modular Steps | ❌ | ✅ |
| Backward Compatible | N/A | ✅ |

The new system is **recommended** for all new work, but the old system remains **fully supported** for existing workflows.
