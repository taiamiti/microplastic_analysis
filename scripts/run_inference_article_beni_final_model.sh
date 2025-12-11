#!/bin/bash
# Run inference on benitiers lots using the BENI_HAO_MAK_TUB model
# This script reproduces the inference for benitiers samples from specific lots

set -e  # Exit on error

# Configuration
CONFIG=fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test
CHECKPOINT=work_dirs/$CONFIG/best_mIoU_iter_2800.pth

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH=mmsegmentation:$PWD

echo "========================================"
echo "Running inference with config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"
echo

# Define lots to process
LOTS=(
    "lot3-08-06-2023-benitiers"
    "lot7-28-09-2023-benitiers"
    "lot8-28-09-2023-benitiers"
    "lot9-09-10-2023-benitiers"
    "lot10-09-10-2023-benitiers"
)

# Process each lot
for DATA in "${LOTS[@]}"; do
    echo "----------------------------------------"
    echo "Processing: $DATA"
    echo "----------------------------------------"

    IMG_FOLDER=data/processed/create_composite/$DATA/data
    SAVE_FOLDER=work_dirs/$CONFIG/inference/$DATA

    # Check if input folder exists
    if [ ! -d "$IMG_FOLDER" ]; then
        echo "Warning: Input folder not found: $IMG_FOLDER"
        echo "Skipping..."
        echo
        continue
    fi

    # Create output folder
    mkdir -p "$SAVE_FOLDER"

    # Run inference
    python mmsegmentation/tools/inference.py \
        --model_cfg mmsegmentation/projects/microplastic_detection/configs/$CONFIG.py \
        --model_ckpts "$CHECKPOINT" \
        --img_folder "$IMG_FOLDER" \
        --save_folder "$SAVE_FOLDER"

    echo "✓ Completed: $DATA"
    echo "  Output: $SAVE_FOLDER"
    echo
done

echo "========================================"
echo "All inference jobs completed!"
echo "========================================"
echo
echo "Output directory: work_dirs/$CONFIG/inference/"
echo
echo "To evaluate results, run the 5-step evaluation workflow:"
echo ""
echo "# Step 1: Add predictions"
echo "python src/modeling/add_predictions.py \\"
echo "  --dataset_name mp_article_beni \\"
echo "  --predictions_dir work_dirs/$CONFIG/inference \\"
echo "  --config_name $CONFIG.py"
echo ""
echo "# Step 2: Evaluate semantic segmentation"
echo "python src/modeling/evaluate_sem_seg.py \\"
echo "  --dataset_name mp_article_beni \\"
echo "  --pred_field predictions_400x400_beni_hao_mak_tub \\"
echo "  --filter_tags test"
echo ""
echo "# Step 3: Convert ground truth to instances (once)"
echo "python src/modeling/convert_to_instance_segmentation.py convert_dataset \\"
echo "  --dataset_name mp_article_beni \\"
echo "  --mask_field ground_truth \\"
echo "  --det_field inst_ground_truth"
echo ""
echo "# Step 4: Convert predictions to instances"
echo "python src/modeling/convert_to_instance_segmentation.py convert_dataset \\"
echo "  --dataset_name mp_article_beni \\"
echo "  --mask_field predictions_400x400_beni_hao_mak_tub \\"
echo "  --det_field inst_predictions_400x400_beni_hao_mak_tub"
echo ""
echo "# Step 5: Evaluate instance segmentation"
echo "python src/modeling/evaluate_inst_seg.py eval_instances \\"
echo "  --dataset_name mp_article_beni \\"
echo "  --pred_field inst_predictions_400x400_beni_hao_mak_tub \\"
echo "  --gt_field inst_ground_truth \\"
echo "  --filter_tags test"
