#!/usr/bin/env python
"""Test script for the new evaluate_sem_seg.py workflow."""

import fiftyone as fo
from src.modeling.eval_utils import extract_config_name

print("="*60)
print("TEST: extract_config_name()")
print("="*60)

# Test config name extraction
test_configs = [
    "fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py",
    "fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_sed_intra_inter_ile.py",
    "400x400_beni_hao_mak_tub.py",
    "mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py"
]

for config in test_configs:
    clean_name = extract_config_name(config)
    print(f"  {config}")
    print(f"  -> {clean_name}\n")

print("\n" + "="*60)
print("TEST: Check if mp_dataset exists")
print("="*60)

if "mp_dataset" in fo.list_datasets():
    dataset = fo.load_dataset("mp_dataset")
    print(f"✓ Dataset 'mp_dataset' found with {len(dataset)} samples")

    # Check for test samples
    test_view = dataset.match_tags("test")
    print(f"✓ Found {len(test_view)} samples with 'test' tag")

    # Check for ground truth
    gt_view = dataset.exists("ground_truth")
    print(f"✓ Found {len(gt_view)} samples with 'ground_truth' field")

    # Show sample structure
    if len(dataset) > 0:
        sample = dataset.first()
        print("\nSample fields:")
        for field in sorted(sample.field_names):
            print(f"  - {field}: {type(sample[field]).__name__}")

        print(f"\nSample tags: {sample.tags}")
        print(f"Sample filepath: {sample.filepath}")

        # Check if any predictions already exist
        pred_fields = [f for f in sample.field_names if f.startswith("predictions_")]
        if pred_fields:
            print(f"\n⚠ Existing prediction fields found: {pred_fields}")

        # Check if any instance fields exist
        inst_fields = [f for f in sample.field_names if f.startswith("inst_")]
        if inst_fields:
            print(f"⚠ Existing instance fields found: {inst_fields}")

    print("\n" + "="*60)
    print("DATASET READY FOR TESTING")
    print("="*60)
    print("\nTo test the semantic evaluation workflow, run:")
    print("")
    print("# Step 1: Add predictions")
    print("python src/modeling/add_predictions.py \\")
    print("  --dataset_name mp_dataset \\")
    print("  --predictions_dir <your_predictions_dir> \\")
    print("  --config_name <your_config>.py")
    print("")
    print("# Step 2: Evaluate semantic segmentation")
    print("python src/modeling/evaluate_sem_seg.py \\")
    print("  --dataset_name mp_dataset \\")
    print("  --pred_field predictions_<config_short_name> \\")
    print("  --filter_tags test")

else:
    print("✗ Dataset 'mp_dataset' not found")
    print("\nAvailable datasets:")
    for ds_name in fo.list_datasets():
        print(f"  - {ds_name}")
    print("\nPlease create 'mp_dataset' first or update the test script with correct dataset name")
