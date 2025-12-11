#!/usr/bin/env python
# coding: utf-8
"""
Convert semantic segmentation masks to instance segmentation with detections.

This script adds instance segmentation to FiftyOne datasets:
- detections: FiftyOne Detections with bounding boxes and instance masks

Each instance includes MP-VAT shape descriptors:
- area, perimeter
- feret_diameter, feret_degree
- circularity, roundness
- mp_shape (Fibers, Fragments, Particles)
- score (contrast-based quality metric)
- mean RGB values

IMPORTANT: For prediction fields, confidence scores are automatically set based on
contrast-based quality metrics (required for mAP computation). If compute_scores=True,
the contrast score is used as confidence; otherwise defaults to 1.0. The script
auto-detects prediction fields by checking if the mask_field name starts with "prediction".
"""

import os
import cv2
import numpy as np
import fiftyone as fo
from tqdm import tqdm
from typing import Optional, Tuple, List
import fire

# Import existing conversion functions
from src.labkit_labeling.generate_annotated_dataset import (
    Detection,
    mp_act,
    filter_dets,
    dets_to_fodetections,
    compute_bg_color,
    compute_box_contrast,
    compute_box_rgb
)


def mask_to_instances(
    mask_path: str,
    img_path: Optional[str] = None,
    compute_scores: bool = True,
    min_area: int = 40,
    max_area: int = 160000
) -> Tuple[List[Detection], Optional[np.ndarray], np.ndarray]:
    """
    Convert semantic mask to Detection objects with optional scoring.

    Args:
        mask_path: Path to binary mask image
        img_path: Optional path to original image for score computation
        compute_scores: Whether to compute contrast scores
        min_area: Minimum area filter
        max_area: Maximum area filter

    Returns:
        Tuple of (detections_list, image_array or None, mask_array)
    """
    # Read mask
    bw = 255 * cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Add black border to avoid issues with connected components at the border
    bw[0, :] = 0
    bw[-1, :] = 0
    bw[:, 0] = 0
    bw[:, -1] = 0

    # Find contours and create Detection objects
    dets_ = mp_act(bw)
    dets = filter_dets(dets_, min_area=min_area, max_area=max_area)

    # Load image if score computation requested
    img = None
    if compute_scores and img_path and os.path.exists(img_path):
        img = cv2.imread(img_path)

    return dets, img, bw


def convert_sample_to_instances(
    sample: fo.Sample,
    mask_field: str = "ground_truth",
    compute_scores: bool = True,
    min_area: int = 40,
    max_area: int = 160000,
    is_prediction: bool = False
) -> fo.Detections:
    """
    Convert a single sample's semantic mask to instance segmentation.

    Args:
        sample: FiftyOne sample with semantic segmentation
        mask_field: Field containing the mask
        compute_scores: Whether to compute contrast scores
        min_area: Minimum area filter
        max_area: Maximum area filter
        is_prediction: Whether this is a prediction (sets confidence=1.0) or ground truth

    Returns:
        FiftyOne Detections with instance masks and MP-VAT attributes
    """
    # Get mask path
    mask_path = sample[mask_field]["mask_path"]
    img_path = sample.filepath

    # Convert mask to instances
    dets, img, mask = mask_to_instances(
        mask_path,
        img_path if compute_scores else None,
        compute_scores,
        min_area,
        max_area
    )

    # Get image dimensions
    img_height, img_width = mask.shape[:2]

    # Convert to FiftyOne Detections
    fo_detections = dets_to_fodetections(dets, img_height, img_width)

    # Compute scores and set confidence if requested
    if compute_scores and img is not None:
        # Resize mask to image size if needed
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Compute background color
        bg_color = compute_bg_color(img, mask)

        # Compute color distance
        L = np.sqrt(np.sum((bg_color - img) ** 2, axis=2))

        # Add scores and RGB values to each detection
        for det in fo_detections.detections:
            # Compute contrast score
            contrast_score = compute_box_contrast(L, det.bounding_box, det.mask)
            det.score = contrast_score

            # For predictions, use contrast score as confidence (required for mAP)
            if is_prediction:
                det.confidence = contrast_score

            # Compute mean RGB values
            b, g, r = compute_box_rgb(img, det.bounding_box, det.mask)
            det.mean_blue = r
            det.mean_green = g
            det.mean_red = b
    elif is_prediction:
        # If scores not computed but this is a prediction, set default confidence
        for det in fo_detections.detections:
            det.confidence = 1.0

    return fo_detections


def add_instance_segmentation_to_dataset(
    dataset: fo.Dataset,
    mask_field: str = "ground_truth",
    det_field: str = "detections",
    compute_scores: bool = True,
    min_area: int = 40,
    max_area: int = 160000,
    batch_size: int = 100,
    filter_tags: Optional[str] = None
) -> fo.Dataset:
    """
    Add instance segmentation detections to all samples in a FiftyOne dataset.

    Args:
        dataset: FiftyOne dataset with semantic segmentation masks
        mask_field: Field containing semantic segmentation (e.g., "ground_truth", "prediction")
        det_field: Output field name for detections
        compute_scores: Whether to compute contrast scores and RGB values
        min_area: Minimum area threshold for filtering detections
        max_area: Maximum area threshold for filtering detections
        batch_size: Number of samples to process before saving
        filter_tags: Optional tag to filter samples (e.g., 'test')

    Returns:
        Updated FiftyOne dataset with instance segmentation detections

    Note:
        If mask_field starts with "prediction", confidence scores will be automatically
        set for all detections (required for mAP computation). With compute_scores=True,
        the contrast-based quality score is used as confidence. Otherwise, defaults to 1.0.
    """
    # Filter samples that have the mask field
    view = dataset.exists(mask_field)

    # Filter by tags if specified
    if filter_tags:
        view = view.match_tags(filter_tags)
        print(f"Filtering to samples with tag: {filter_tags}")

    print(f"Converting {len(view)} samples from {mask_field} to {det_field}...")

    # Auto-detect if this is prediction field (requires confidence scores)
    is_prediction = mask_field.startswith("prediction")
    if is_prediction:
        if compute_scores:
            print("Detected prediction field - using contrast score as confidence")
        else:
            print("Detected prediction field - setting confidence=1.0 for all detections")

    # Process in batches for efficiency
    with dataset.save_context() as context:
        for sample in tqdm(view, desc="Converting to instances"):
            try:
                # Convert sample to instances
                detections = convert_sample_to_instances(
                    sample,
                    mask_field=mask_field,
                    compute_scores=compute_scores,
                    min_area=min_area,
                    max_area=max_area,
                    is_prediction=is_prediction
                )

                # Add detections to sample
                sample[det_field] = detections

                # Save sample
                context.save(sample)

            except Exception as e:
                print(f"Warning: Failed to process sample {sample.filepath}: {e}")
                continue

    # Add dynamic fields and save
    dataset.add_dynamic_sample_fields()
    dataset.save()

    print(f"Successfully added {det_field} to {len(view)} samples")

    return dataset


def convert_dataset_cli(
    dataset_name: str,
    mask_field: str = "ground_truth",
    det_field: str = "detections",
    compute_scores: bool = True,
    min_area: int = 40,
    max_area: int = 160000,
    batch_size: int = 100,
    filter_tags: Optional[str] = None
):
    """
    CLI wrapper: Load a FiftyOne dataset by name and add instance segmentation.

    Args:
        dataset_name: Name of the FiftyOne dataset to load
        mask_field: Field containing semantic segmentation (e.g., "ground_truth", "prediction")
        det_field: Output field name for detections
        compute_scores: Whether to compute contrast scores and RGB values
        min_area: Minimum area threshold for filtering detections
        max_area: Maximum area threshold for filtering detections
        batch_size: Number of samples to process before saving
        filter_tags: Optional tag to filter samples (e.g., 'test')

    Note:
        If mask_field starts with "prediction", confidence scores will be automatically
        set for all detections (required for mAP computation). With compute_scores=True,
        the contrast-based quality score is used as confidence. Otherwise, defaults to 1.0.
    """
    # Load dataset
    dataset = fo.load_dataset(dataset_name)

    # Process dataset
    add_instance_segmentation_to_dataset(
        dataset=dataset,
        mask_field=mask_field,
        det_field=det_field,
        compute_scores=compute_scores,
        min_area=min_area,
        max_area=max_area,
        batch_size=batch_size,
        filter_tags=filter_tags
    )

    print(f"Dataset '{dataset_name}' updated successfully!")


if __name__ == '__main__':
    fire.Fire(convert_dataset_cli)
