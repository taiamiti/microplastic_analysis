#!/usr/bin/env python
"""
Shared utility functions for evaluation scripts.

This module provides common utilities used by semantic and instance
segmentation evaluation workflows.
"""

from pathlib import Path


def extract_config_name(config_path: str) -> str:
    """
    Extract clean config name from full path.

    Removes common prefixes to create a shortened name suitable for
    FiftyOne field naming.

    Example:
        >>> extract_config_name('fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py')
        '400x400_train_test'

        >>> extract_config_name('mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_sed_intra_inter_ile.py')
        '256x256_sed_intra_inter_ile'

    Args:
        config_path: Path to config file or just the filename

    Returns:
        Shortened config name for field naming
    """
    # Get filename without extension
    filename = Path(config_path).stem

    # Remove common prefixes/suffixes to shorten
    name = filename.replace('fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-', '')

    return name
