#!/usr/bin/env python
"""
Instance segmentation evaluation script.

This script provides CLI commands for:
1. Evaluating instance segmentation metrics (mAP, per-detection precision/recall)

NOTE: This script does NOT include instance conversion. Users should use the existing
`convert_to_instance_segmentation.py` script to convert semantic masks to instance
detections before running evaluation.

IMPORTANT: Predicted detections MUST have confidence scores populated. The conversion
script automatically sets confidence scores when mask_field starts with "prediction".
With compute_scores=True (default), it uses contrast-based quality scores as confidence;
otherwise it defaults to 1.0.

Field naming convention:
- Instance predictions: inst_predictions_{config_name}
- Instance GT: inst_ground_truth
- Evaluation keys: eval_inst_{config_name}

Example workflow:
    # Step 1: Convert ground truth to instances (once per dataset)
    python src/modeling/convert_to_instance_segmentation.py convert_dataset \\
        --dataset_name mp_dataset \\
        --mask_field ground_truth \\
        --det_field inst_ground_truth

    # Step 2: Convert predictions to instances (auto-sets confidence=contrast_score)
    python src/modeling/convert_to_instance_segmentation.py convert_dataset \\
        --dataset_name mp_dataset \\
        --mask_field predictions_400x400_train_test \\
        --det_field inst_predictions_400x400_train_test

    # Step 3: Evaluate instance segmentation (class-aware by default)
    python src/modeling/evaluate_inst_seg.py \\
        --dataset_name mp_dataset \\
        --pred_field inst_predictions_400x400_train_test \\
        --gt_field inst_ground_truth \\
        --filter_tags test

    # Optional: Class-agnostic evaluation (matches objects regardless of class)
    python src/modeling/evaluate_inst_seg.py \\
        --dataset_name mp_dataset \\
        --pred_field inst_predictions_400x400_train_test \\
        --gt_field inst_ground_truth \\
        --filter_tags test \\
        --classwise False
"""

from typing import Optional, Dict, Any
import fiftyone as fo
import fire


def eval_instances(
    dataset_name: str,
    pred_field: str,
    gt_field: str,
    eval_key: Optional[str] = None,
    filter_tags: Optional[str] = "test",
    iou_threshold: float = 0.5,
    compute_mAP: bool = True,
    classwise: bool = False
) -> Dict[str, Any]:
    """
    Evaluate instance segmentation metrics.

    Args:
        dataset_name: Name of FiftyOne dataset
        pred_field: Field containing predicted detections (e.g., 'inst_predictions_400x400_train_test')
        gt_field: Field containing ground truth detections (e.g., 'inst_ground_truth')
        eval_key: Optional evaluation key. If None, uses 'eval_{pred_field}'
        filter_tags: Tag to filter samples (default: 'test')
        iou_threshold: IoU threshold for evaluation (default: 0.5)
        compute_mAP: Whether to compute mAP metrics (default: True)
        classwise: Whether to only match objects with the same class label (default: False).
                   Set to False for class-agnostic evaluation.

    Returns:
        Evaluation results dictionary

    Example:
        >>> # Class-aware evaluation (default)
        >>> eval_instances(
        ...     dataset_name='mp_dataset',
        ...     pred_field='inst_predictions_400x400_train_test',
        ...     gt_field='inst_ground_truth',
        ...     filter_tags='test'
        ... )
        >>>
        >>> # Class-agnostic evaluation
        >>> eval_instances(
        ...     dataset_name='mp_dataset',
        ...     pred_field='inst_predictions_400x400_train_test',
        ...     gt_field='inst_ground_truth',
        ...     filter_tags='test',
        ...     classwise=False
        ... )
    """
    # Load dataset
    dataset = fo.load_dataset(dataset_name)

    # Default eval key
    if eval_key is None:
        eval_key = f"eval_{pred_field.replace('inst_predictions_', 'inst_')}"

    print(f"Dataset: {dataset_name}")
    print(f"Prediction field: {pred_field}")
    print(f"Ground truth field: {gt_field}")
    print(f"Evaluation key: {eval_key}")
    print(f"Evaluation mode: {'Class-aware' if classwise else 'Class-agnostic'}")

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
    results = view.evaluate_detections(
        pred_field,
        gt_field=gt_field,
        eval_key=eval_key,
        compute_mAP=compute_mAP, # if true, computes in a separate layer the coco style iou sweep mAP
        iou=iou_threshold,  # set this to a single value (default = 0.5)
        classwise=False,  # this allows us to see confusion matrix (class agnostic matching)
        use_masks=True, # iou is computed using masks instead of bbox (slower)
    )

    # Print results
    print("\n" + "="*60)
    print("INSTANCE SEGMENTATION RESULTS")
    print("="*60)

    print(f"\nmAP: {results.mAP():.4f}")

    # Per-class metrics
    print("\nPer-class metrics:")
    print(results.report())
    return results


if __name__ == '__main__':
    fire.Fire(eval_instances)
