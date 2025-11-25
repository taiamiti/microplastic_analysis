# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project automates the analysis of microplastic contamination from fluorescent microscopy images. It processes microscopy data through a complete pipeline: data ingestion, annotation using Labkit interactive learning, model training via mmsegmentation, and CSV export for analysis.

Data naming convention: `{sample_type}_{island}_{station}_{replica}_{distil}_{sample_id}_{filter}`

## Environment Setup

**Two separate environments are required:**

### 1. Data Engineering Environment (map_de)
For data preparation, annotation, and export:
```bash
conda create -n map_de python=3.9
conda activate map_de
pip install -r requirements_de.txt
export PYTHONPATH=$PWD
```

### 2. Modeling Environment (openmmlab)
For training and inference with mmsegmentation:
```bash
# Follow installation instructions in mmsegmentation/README.md
conda activate openmmlab
export PYTHONPATH=mmsegmentation:$PWD
```

**Important:** These environments must be kept separate due to conflicting torch dependencies between CLIP embeddings (data engineering) and mmsegmentation (modeling).

## Common Commands

### Data Preparation Pipeline

```bash
# Always set PYTHONPATH first
export PYTHONPATH=$PWD
conda activate map_de

# Ingest raw data (renames files, infers filters, validates acquisitions)
python src/pipeline.py ingest_data_subset configs/default_config.yaml <lot-name>
python src/pipeline.py ingest_data configs/default_config.yaml  # all lots

# Create composite images from 4-channel microscopy data
python src/pipeline.py create_composite_subset configs/default_config.yaml <lot-name>
python src/pipeline.py create_composite configs/default_config.yaml  # all lots

# Generate annotated dataset (pairs images with masks)
python src/pipeline.py generate_annotated_subset configs/default_config.yaml <lot-name>
python src/pipeline.py generate_annotated_dataset configs/default_config.yaml  # all lots

# Prepare dataset for mmsegmentation training
python src/pipeline.py prepare_dataset_for_openmmseg configs/default_config.yaml
```

### Model Training and Inference

```bash
conda activate openmmlab
export PYTHONPATH=mmsegmentation:$PWD

# Train model
python mmsegmentation/tools/train.py \
  mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py \
  --work-dir data/modeling/work_dirs

# Run inference on unlabeled data
python mmsegmentation/tools/inference.py \
  --model_config <config.py> \
  --model_ckpts <checkpoint.pth> \
  --img_folder <input_folder> \
  --save_folder <output_folder>

# Evaluate with FiftyOne
conda activate map_de
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  <inference_output_dir> \
  --eval_bool True
```

### Export Results to CSV

```bash
conda activate map_de

# Export specific unlabeled lot with predictions
python src/pipeline.py export_unlabelled_folder configs/default_config.yaml \
  data/processed/create_composite/<lot-name> \
  <inference_output_dir>

# Export all annotated datasets
python src/pipeline.py export configs/default_config.yaml <inference_root_dir>
```

## Code Architecture

### Module Structure

- **data_prep/**: Raw data ingestion and composite creation
  - `ingest_data.py`: Renames files, infers filter types using CLIP embeddings, validates zoom/size
  - `create_composite.py`: Creates composite images from 4-channel microscopy (NAT, DAPI, TRI, CY2)
  - `embeddings.py`: CLIP-based filter type inference using pre-computed embedding centers

- **labkit_labeling/**: Annotation workflow management
  - `create_tasks.py`: Groups images into annotation tasks using clustering
  - `matching_old_names_with_new.py`: Reorganizes Labkit outputs back to original structure
  - `generate_annotated_dataset.py`: Pairs images with masks, tags train/test splits
  - `prepare_dataset_for_openmmseg.py`: Converts FiftyOne dataset to ImageSequence format

- **modeling/**: Model evaluation using FiftyOne
  - `run_fiftyone_eval.py`: Visualizes predictions vs ground truth in FiftyOne UI

- **export/**: Instance detection and CSV export
  - `exporter.py`: Converts masks to instance detections with shape descriptors (similar to MP-VAT2.0 format)

- **viz/**: Jupyter notebooks for dataset visualization

- **mmsegmentation/projects/microplastic_detection/**: Custom mmsegmentation project
  - Custom transforms: `InvertBinaryLabels`, `RandomCropForeground`
  - Custom dataset: `MicroPlasticDataset`
  - Training configs for different protocols and input sizes

### Pipeline Entry Point

`src/pipeline.py` uses Fire CLI to expose all pipeline steps. Each method loads `configs/default_config.yaml` and processes data through standardized folder paths defined in the config.

### Configuration

`configs/default_config.yaml` defines:
- Data folder structure (raw → ingested → composite → annotated → trainval)
- Modeling experiment directory
- Sampling parameters for annotation tasks (clustering, sampling rates)
- Embedding centers path for filter type inference

## Data Organization

Raw data is organized into 11 acquisition campaign lots, sometimes split into parts to maintain naming consistency. Three naming conventions exist historically:
- RENAMED: Fully consistent naming from start
- PARTIAL_RENAMED: Only observation IDs renamed
- CONSECUTIVE: 4 consecutive images per observation

The pipeline handles all three conventions to group images into 4-image observations (one per filter).

## Annotation Workflow (Labkit)

1. Create annotation tasks: `create_tasks` groups images via clustering to limit variability
2. Manual annotation in Labkit ImageJ plugin (see tutorial: https://docs.google.com/presentation/d/12bUywRMCjIyrB3BmrCNps7Y_XApCsKtEKgkffKQYOjs/edit#slide=id.p)
3. Run Labkit inference using ImageJ macros in `src/labkit_labeling/labkitmacro_*.ijm`
4. Reorganize masks: `matching_old_names_with_new` restores original folder structure
5. Generate annotated dataset: pairs images with masks in FiftyOne format

**Note:** Labkit models are saved per annotation task under `data/processed/labkit_models`

## Train/Test Protocols

- **train_test**: General train/test split using 30% of data, split by sample origin
- **sed_intra_inter_ile**: Sediment-specific protocol with intra/inter island splits

## Key Implementation Details

- Filter type inference uses CLIP embeddings matched to pre-computed centers from lot2
- Only zoom=500/200 with size=1920x1200 are kept as valid acquisitions
- Composite images combine 4 filters: NAT (natural light), DAPI, TRI, CY2
- When both Labkit GT and model predictions exist, exports use GT annotations
- Custom mmsegmentation transforms handle foreground cropping and label inversion
