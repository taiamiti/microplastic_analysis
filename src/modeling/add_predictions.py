#!/usr/bin/env python
"""
Add prediction masks to FiftyOne datasets.

This script provides a CLI command for adding semantic segmentation prediction masks
to existing FiftyOne datasets with automatic field naming based on config files.

Field naming convention:
- Semantic predictions: predictions_{config_name}

Example usage:
    python src/modeling/add_predictions.py \
        --dataset_name mp_dataset \
        --predictions_dir data/modeling/work_dirs/unet_400/inference \
        --config_name fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py
"""

import os
from pathlib import Path
from typing import Optional
import fiftyone as fo
import fire

from src.modeling.eval_utils import extract_config_name


def add_predictions(
    dataset_name: str,
    predictions_dir: str,
    config_name: str,
    pred_field: Optional[str] = None,
    filter_tags: Optional[str] = None,
    label_ext: str = ".png"
) -> fo.Dataset:
    """
    Add prediction masks to an existing FiftyOne dataset.

    Args:
        dataset_name: Name of existing persistent FiftyOne dataset
        predictions_dir: Root directory containing prediction masks organized by lot
        config_name: Config filename (with or without .py) for unique field naming
        pred_field: Optional custom field name. If None, uses 'predictions_{config_name}'
        filter_tags: Optional tag to filter samples (e.g., 'test')
        label_ext: Extension for mask files (default: '.png')

    Returns:
        Updated FiftyOne dataset

    Example:
        >>> add_predictions(
        ...     dataset_name='mp_dataset',
        ...     predictions_dir='data/modeling/work_dirs/unet_400x400/inference',
        ...     config_name='fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py'
        ... )
    """
    # Load dataset
    dataset = fo.load_dataset(dataset_name)

    # Extract clean config name
    clean_config_name = extract_config_name(config_name)

    # Determine field name
    if pred_field is None:
        pred_field = f"predictions_{clean_config_name}"

    print(f"Loading dataset: {dataset_name}")
    print(f"Config name: {config_name}")
    print(f"Predictions field: {pred_field}")
    print(f"Predictions directory: {predictions_dir}")

    # Filter samples if tag specified
    view = dataset
    if filter_tags:
        view = dataset.match_tags(filter_tags)
        print(f"Filtering to samples with tag: {filter_tags}")

    print(f"Processing {len(view)} samples...")

    # Add predictions
    added_count = 0
    missing_count = 0

    for sample in view.select_fields("filepath").iter_samples(autosave=True, progress=True):
        p = Path(sample.filepath)

        # Extract lot name from path structure
        # Expected structure: .../lot_name/data/image.jpg
        lot = p.parts[-3]
        img_name = p.parts[-1]

        # Build mask path
        mask_rel_path = os.path.join(lot, img_name.replace(".jpg", label_ext))
        mask_path = os.path.abspath(os.path.join(predictions_dir, mask_rel_path))

        if os.path.exists(mask_path):
            sample[pred_field] = fo.Segmentation(mask_path=mask_path)
            added_count += 1
        else:
            missing_count += 1

    # Update dynamic fields
    dataset.add_dynamic_sample_fields()
    dataset.save()

    print(f"\n✓ Added predictions to {added_count} samples")
    if missing_count > 0:
        print(f"⚠ Warning: {missing_count} samples missing prediction masks")

    return dataset


if __name__ == '__main__':
    fire.Fire(add_predictions)
