# Changelog: New Evaluation System

**Date:** 2025-11-25
**Author:** Claude Code
**Status:** Ready for testing

## Summary

Refactored the FiftyOne evaluation workflow to support both semantic and instance segmentation evaluation with config-based field naming for multi-model comparison.

## New Files Created

### 1. `src/modeling/evaluate_segmentation.py` (17KB, 580 lines)

Main evaluation script with 5 CLI commands:

- `add_predictions` - Add predictions to existing dataset
- `eval_semantic` - Evaluate semantic segmentation (IoU, Dice, etc.)
- `convert_to_instances` - Convert semantic masks to instance detections
- `eval_instances` - Evaluate instance segmentation (mAP, precision, recall)
- `run_full_eval` - All-in-one pipeline

**Key Features:**
- Config-based field naming (e.g., `predictions_400x400_train_test`)
- Supports multiple models on same dataset without conflicts
- Modular steps for debugging and flexibility
- Automatic instance conversion with MP-VAT descriptors
- Full evaluation metrics for both semantic and instance segmentation

### 2. `test_evaluate_segmentation.py` (2.6KB)

Test script to verify:
- Config name extraction works correctly
- Dataset structure is valid
- Fields are created properly
- Ready for testing the full workflow

### 3. `docs/EVALUATION_WORKFLOW.md` (12KB)

Comprehensive documentation with:
- CLI command reference for all functions
- Field naming conventions
- Complete examples (step-by-step and all-in-one)
- Multi-model comparison workflow
- Troubleshooting guide
- FiftyOne App visualization tips

### 4. `docs/MIGRATION_GUIDE.md` (6KB)

Migration guide covering:
- Old vs new workflow comparison
- Step-by-step migration instructions
- Backward compatibility notes
- Example transformations
- Common issues and solutions

## Updated Files

### `README.md`

**Changes:**
- Added "Quick Start: Evaluation Workflow" section at the top
- Updated Step 8.3 with new evaluation system (marked as recommended)
- Added Step 8.4 for multi-model comparison
- Reorganized Step 8.5 (save dataset) and Step 8.6 (instance conversion)
- Added overview of modeling workflow with new system
- Kept old workflow documentation for backward compatibility

**Key Sections:**
- Lines 25-38: Quick start section
- Lines 278-289: Modeling overview
- Lines 415-495: Complete evaluation workflow documentation
- Lines 497-532: Multi-model comparison example

## Field Naming Convention

### Semantic Segmentation
- **Predictions**: `predictions_{config_name}`
- **Evaluation**: `eval_{config_name}`
- **Example**: `predictions_400x400_train_test`

### Instance Segmentation
- **Ground Truth**: `inst_ground_truth` (created once)
- **Predictions**: `inst_predictions_{config_name}`
- **Evaluation**: `eval_inst_{config_name}`
- **Example**: `inst_predictions_400x400_train_test`

### Config Name Extraction

Long config names are automatically shortened:

```
Input:  fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py
Output: 400x400_train_test
```

## Usage Examples

### All-in-One Evaluation

```bash
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model/inference \
  --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py \
  --filter_tags test
```

### Step-by-Step Evaluation

```bash
# 1. Add predictions
python src/modeling/evaluate_segmentation.py add_predictions \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model/inference \
  --config_name your_config.py

# 2. Evaluate semantic
python src/modeling/evaluate_segmentation.py eval_semantic \
  --dataset_name mp_dataset \
  --pred_field predictions_your_config

# 3. Convert to instances
python src/modeling/evaluate_segmentation.py convert_to_instances \
  --dataset_name mp_dataset \
  --mask_field predictions_your_config

# 4. Evaluate instances
python src/modeling/evaluate_segmentation.py eval_instances \
  --dataset_name mp_dataset \
  --pred_field inst_predictions_your_config
```

### Multi-Model Comparison

```bash
# Evaluate Model 1
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir work_dirs/model1/inference \
  --config_name config1.py

# Evaluate Model 2
python src/modeling/evaluate_segmentation.py run_full_eval \
  --dataset_name mp_dataset \
  --predictions_dir work_dirs/model2/inference \
  --config_name config2.py

# Compare in FiftyOne App
python -c "import fiftyone as fo; fo.launch_app(fo.load_dataset('mp_dataset').match_tags('test'))"
```

## Benefits

### For Users
✅ **Easier model comparison** - Evaluate multiple models on same dataset
✅ **Complete metrics** - Both semantic (IoU, Dice) and instance (mAP) evaluation
✅ **Persistent datasets** - No need to recreate dataset each time
✅ **Clear naming** - Config-based fields make it obvious which model is which
✅ **Modular workflow** - Run individual steps for debugging

### For Development
✅ **Reuses existing code** - Leverages `convert_to_instance_segmentation.py`
✅ **Backward compatible** - Old workflow still works
✅ **Well documented** - Comprehensive docs and examples
✅ **Testable** - Includes test script

## Testing Checklist

Before using in production:

- [ ] Test config name extraction
  ```bash
  python src/modeling/evaluate_segmentation.py extract_config_name "your_config.py"
  ```

- [ ] Verify dataset exists
  ```bash
  python test_evaluate_segmentation.py
  ```

- [ ] Test adding predictions
  ```bash
  python src/modeling/evaluate_segmentation.py add_predictions \
    --dataset_name mp_dataset \
    --predictions_dir path/to/inference \
    --config_name your_config.py
  ```

- [ ] Test semantic evaluation
  ```bash
  python src/modeling/evaluate_segmentation.py eval_semantic \
    --dataset_name mp_dataset \
    --pred_field predictions_your_config
  ```

- [ ] Test instance conversion
  ```bash
  python src/modeling/evaluate_segmentation.py convert_to_instances \
    --dataset_name mp_dataset \
    --mask_field ground_truth
  ```

- [ ] Test instance evaluation
  ```bash
  python src/modeling/evaluate_segmentation.py eval_instances \
    --dataset_name mp_dataset \
    --pred_field inst_predictions_your_config
  ```

- [ ] Test full pipeline
  ```bash
  python src/modeling/evaluate_segmentation.py run_full_eval \
    --dataset_name mp_dataset \
    --predictions_dir path/to/inference \
    --config_name your_config.py
  ```

- [ ] Verify in FiftyOne App
  ```python
  import fiftyone as fo
  dataset = fo.load_dataset("mp_dataset")
  session = fo.launch_app(dataset)
  # Check that fields are created correctly
  ```

## Backward Compatibility

The old `run_fiftyone_eval.py` script is **fully supported** and unchanged. Users can continue using it if they:
- Prefer the monolithic workflow
- Only need semantic segmentation
- Don't need multi-model comparison

## Next Steps

1. **Test the workflow** with your actual models and data
2. **Review the documentation** in `docs/EVALUATION_WORKFLOW.md`
3. **Try multi-model comparison** to evaluate different experiments
4. **Provide feedback** on the field naming convention
5. **Report any issues** or missing features

## Related Files

- **Main script**: `src/modeling/evaluate_segmentation.py`
- **Test script**: `test_evaluate_segmentation.py`
- **Documentation**: `docs/EVALUATION_WORKFLOW.md`
- **Migration guide**: `docs/MIGRATION_GUIDE.md`
- **Updated README**: `README.md` (Steps 8.3, 8.4, 8.6)
- **Instance conversion**: `src/modeling/convert_to_instance_segmentation.py` (reused)
- **Legacy script**: `src/modeling/run_fiftyone_eval.py` (unchanged)

## Technical Details

### Dependencies
- FiftyOne (existing)
- Fire CLI (existing)
- All existing conversion utilities from `convert_to_instance_segmentation.py`

### Python API
All functions can also be imported and used from Python:

```python
from src.modeling.evaluate_segmentation import (
    add_predictions,
    eval_semantic,
    convert_to_instances,
    eval_instances,
    run_full_eval
)

# Use functions programmatically
dataset = add_predictions(
    dataset_name="mp_dataset",
    predictions_dir="path/to/inference",
    config_name="your_config.py"
)
```

### Error Handling
- Validates dataset exists before operations
- Checks for required fields before evaluation
- Reports missing prediction files with counts
- Handles conversion errors gracefully with warnings

## Performance Notes

- **Batch processing**: Instance conversion saves every 100 samples by default
- **Memory efficient**: Processes samples one at a time
- **Concurrent safe**: Uses FiftyOne's save context for thread safety
- **Incremental**: Can skip already-completed steps (e.g., `--skip_add_predictions`)

## Known Limitations

1. **Config name extraction**: Assumes standard naming convention. Use `--pred_field` for custom names.
2. **Directory structure**: Expects predictions organized by lot (e.g., `lot1/image.png`)
3. **Mask format**: Expects binary PNG masks with 0=background, 255=microplastic
4. **Instance conversion**: Requires source images for score computation (optional)

## Future Enhancements

Potential improvements for future versions:
- Support for additional mask formats (COCO, YOLO, etc.)
- Automated report generation (PDF/HTML)
- Integration with MLflow or W&B for experiment tracking
- Batch evaluation across multiple datasets
- Custom metric plugins
- Interactive comparison dashboard

---

**Status**: Ready for testing and feedback
**Compatibility**: Python 3.9+, FiftyOne 0.20+
**License**: Same as project
