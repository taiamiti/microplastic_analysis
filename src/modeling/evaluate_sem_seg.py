#!/usr/bin/env python
"""
Semantic segmentation evaluation script.

This script provides a CLI command for evaluating semantic segmentation metrics
(IoU, Dice, precision, recall) on FiftyOne datasets.

NOTE: Use the standalone `add_predictions.py` script to add predictions to your
dataset before running evaluation.

Field naming convention:
- Evaluation keys: eval_{config_name}

Example usage:
    # Evaluate semantic segmentation
    python src/modeling/evaluate_sem_seg.py eval_semantic \
        --dataset_name mp_dataset \
        --pred_field predictions_400x400_train_test \
        --filter_tags test
"""

from typing import Optional, Dict, Any
import fiftyone as fo
import fire


def eval_semantic(
    dataset_name: str,
    pred_field: str,
    gt_field: str = "ground_truth",
    eval_key: Optional[str] = None,
    filter_tags: Optional[str] = "test",
    mask_targets: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Evaluate semantic segmentation metrics.

    Args:
        dataset_name: Name of FiftyOne dataset
        pred_field: Field containing predictions (e.g., 'predictions_400x400_train_test')
        gt_field: Field containing ground truth (default: 'ground_truth')
        eval_key: Optional evaluation key. If None, uses 'eval_{pred_field}'
        filter_tags: Tag to filter samples (default: 'test')
        mask_targets: Dict mapping pixel values to class names

    Returns:
        Evaluation results dictionary

    Example:
        >>> eval_semantic(
        ...     dataset_name='mp_dataset',
        ...     pred_field='predictions_400x400_train_test',
        ...     filter_tags='test'
        ... )
    """
    # Load dataset
    dataset = fo.load_dataset(dataset_name)

    # Default eval key
    if eval_key is None:
        eval_key = f"eval_{pred_field.replace('predictions_', '')}"

    # Default mask targets
    if mask_targets is None:
        mask_targets = {0: "background", 255: "mp"}

    print(f"Dataset: {dataset_name}")
    print(f"Prediction field: {pred_field}")
    print(f"Ground truth field: {gt_field}")
    print(f"Evaluation key: {eval_key}")

    # Create view
    view = dataset
    if filter_tags:
        view = dataset.match_tags(filter_tags)
        print(f"Filtering to tag: {filter_tags}")

    # Check both fields exist
    view = view.exists(pred_field).exists(gt_field)
    print(f"Evaluating {len(view)} samples...")

    if len(view) == 0:
        print("⚠ No samples found with both prediction and ground truth fields!")
        return {}

    # Run evaluation
    results = view.evaluate_segmentations(
        pred_field,
        gt_field=gt_field,
        eval_key=eval_key,
        mask_targets=mask_targets
    )

    # Print results
    print("\n" + "="*60)
    print("SEMANTIC SEGMENTATION RESULTS")
    print("="*60)

    # Per-sample statistics
    print(f"\nAccuracy range: ({dataset.bounds(f'{eval_key}_accuracy')[0]:.4f}, {dataset.bounds(f'{eval_key}_accuracy')[1]:.4f})")
    print(f"Precision range: ({dataset.bounds(f'{eval_key}_precision')[0]:.4f}, {dataset.bounds(f'{eval_key}_precision')[1]:.4f})")
    print(f"Recall range: ({dataset.bounds(f'{eval_key}_recall')[0]:.4f}, {dataset.bounds(f'{eval_key}_recall')[1]:.4f})")

    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    results.print_report()

    return results


if __name__ == '__main__':
    fire.Fire(eval_semantic)
