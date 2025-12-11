#!/usr/bin/env python
"""Test script for instance segmentation conversion."""

import fiftyone as fo
from src.modeling.convert_to_instance_segmentation import add_instance_segmentation_to_dataset

# Load test dataset
dataset_path = "data/processed/fiftyone_evaluations/ds_export"
print(f"Loading dataset from: {dataset_path}")

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_path,
    dataset_type=fo.types.FiftyOneDataset,
    name="mp_dataset"
).match_tags("test").take(6).clone("mp_dataset_small")

print(f"Loaded {len(dataset)} samples")

det_field_name = "detections2"

# Run conversion
add_instance_segmentation_to_dataset(
    dataset,
    mask_field="ground_truth",
    det_field=det_field_name,
    compute_scores=True,
    min_area=40,
    max_area=160000
)

# Verify results
print("\n" + "="*60)
print("VERIFICATION RESULTS")
print("="*60)

for i, sample in enumerate(dataset, 1):
    print(f"\nSample {i}: {sample.filepath.split('/')[-1]}")

    if sample.has_field(det_field_name) and sample.detections is not None:
        num_dets = len(sample[det_field_name].detections)
        print(f"  ✓ Detections field exists: {num_dets} instances")

        if num_dets > 0:
            det = sample[det_field_name].detections[0]
            print(f"  First detection attributes:")
            print(f"    - label: {det.label}")
            print(f"    - area: {det.area:.2f}")
            print(f"    - feret_diameter: {det.feret_diameter:.2f}")
            print(f"    - circularity: {det.circularity:.4f}")
            print(f"    - roundness: {det.roundness:.4f}")
            print(f"    - perimeter: {det.perimeter:.2f}")
            if hasattr(det, 'score'):
                print(f"    - score: {det.score:.4f}")
            if hasattr(det, 'mean_red'):
                print(f"    - RGB: ({det.mean_red:.1f}, {det.mean_green:.1f}, {det.mean_blue:.1f})")
    else:
        print(f"  ✗ No detections field found")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

total_instances = sum(len(s[det_field_name].detections) if s.has_field(det_field_name) and s[det_field_name] else 0
                     for s in dataset)
print(f"Total instances detected: {total_instances}")

if total_instances > 0:
    # Count by particle type
    particle_types = {}
    for sample in dataset:
        if sample.has_field(det_field_name) and sample[det_field_name]:
            for det in sample[det_field_name].detections:
                particle_types[det.label] = particle_types.get(det.label, 0) + 1

    print("\nParticle type distribution:")
    for ptype, count in particle_types.items():
        print(f"  {ptype}: {count}")

print("\n✓ Test completed successfully!")

# Cleanup
dataset.delete()
