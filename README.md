# Microplastic analysis

**Date** : 16/05/2023  
**Author** : Taiamiti Edmunds (taiamiti.edmunds@ml4everyone.com)  
**Goals** : This project aims to automate the analysis of microplastic contamination on the environment from 
fluorescent microscopy images

This project contains scripts to :
- process data
- annotate data
- train and evaluate models 
- perform inference on new data

This code is associated to the publication ** and thesis of Irène Godere

## Overview

The code is structured into 5 modules :
- data_prep : scripts to prepare data and create image composite
- labkit_labeling : scripts to analyse, cluster and annotate data
- modeling : training with mmsegmentation + evaluation with FiftyOne (semantic & instance segmentation)
- export : export dataset (fiftyone) with ground truths or predictions to csv
- viz : use notebooks to visualize datasets

### Quick Start: Evaluation Workflow

For evaluating trained models with both semantic and instance segmentation metrics:

```bash
# Step 1: Add predictions
python src/modeling/add_predictions.py \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/model/inference \
  --config_name <your_config>.py

# Step 2: Evaluate semantic segmentation
python src/modeling/evaluate_sem_seg.py \
  --dataset_name mp_dataset \
  --pred_field predictions_<config_short_name> \
  --filter_tags test

# Step 3: Convert ground truth to instances (once per dataset)
python src/modeling/convert_to_instance_segmentation.py convert_dataset \
  --dataset_name mp_dataset \
  --mask_field ground_truth \
  --det_field inst_ground_truth

# Step 4: Convert predictions to instances
python src/modeling/convert_to_instance_segmentation.py convert_dataset \
  --dataset_name mp_dataset \
  --mask_field predictions_<config_short_name> \
  --det_field inst_predictions_<config_short_name>

# Step 5: Evaluate instance segmentation
python src/modeling/evaluate_inst_seg.py \
  --dataset_name mp_dataset \
  --pred_field inst_predictions_<config_short_name> \
  --gt_field inst_ground_truth \
  --filter_tags test
```

See [Step 8.3](#step-83--evaluate-models-semantic-and-instance-segmentation) for detailed instructions and [docs/EVALUATION_WORKFLOW.md](docs/EVALUATION_WORKFLOW.md) for complete documentation.

## Installation

**Important:** This project uses a **customized version** of mmsegmentation located in `./mmsegmentation` with custom datasets, transforms, and inference scripts. The local version is automatically installed - do NOT install mmsegmentation from PyPI.

### Using Pixi (Unified Environment)

Pixi provides a single unified environment for all pipeline tasks (data engineering + modeling):

```bash
# Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies (automatically installs local customized mmsegmentation)
pixi install

# Activate the environment
pixi shell

# Set PYTHONPATH for pipeline scripts
export PYTHONPATH=mmsegmentation:$PWD
```

This installs:
- Python 3.9
- PyTorch 2.0.0 with CUDA 11.7
- mmcv 2.0.1 and mmengine
- Local customized mmsegmentation (editable install)
- All data engineering dependencies (CLIP, FiftyOne, etc.)

## Data naming convention

All files are renamed as follows `{sample_type}_{island}_{station}_{replica}_{distil}_{sample_id}`
At first, this was done manually, but it was automated later due to inconsistencies and waste of time 
Hence, we handled 3 cases before full automation as described below :
- RENAMED = 1  # ex : lot 1 completely renamed manually using the convention
```text
├── BENI_TAK_S1_3V_D0_F1_0000_CY2.jpg
├── BENI_TAK_S1_3V_D0_F1_0000_DAPI.jpg
├── BENI_TAK_S1_3V_D0_F1_0000_NAT.jpg
├── BENI_TAK_S1_3V_D0_F1_0000_TRI.jpg
├── BENI_TAK_S1_3V_D0_F1_0001_CY2.jpg
├── BENI_TAK_S1_3V_D0_F1_0001_DAPI.jpg
├── BENI_TAK_S1_3V_D0_F1_0001_NAT.jpg
├── BENI_TAK_S1_3V_D0_F1_0001_TRI.jpg
...
```
- PARTIAL_RENAMED = 2  # ex lot2 (only obs id renamed)
```text
├── CSED_TAK_S1_UNK_UNK_0001 (1).jpg
├── CSED_TAK_S1_UNK_UNK_0001 (2).jpg
├── CSED_TAK_S1_UNK_UNK_0001 (3).jpg
├── CSED_TAK_S1_UNK_UNK_0001 (4).jpg
├── CSED_TAK_S1_UNK_UNK_0001_bis (3).jpg
├── CSED_TAK_S1_UNK_UNK_0002 (1).jpg
├── CSED_TAK_S1_UNK_UNK_0002 (2).jpg
├── CSED_TAK_S1_UNK_UNK_0002 (3).jpg
├── CSED_TAK_S1_UNK_UNK_0002 (4).jpg
...
```
- CONSECUTIVE = 3  # ex lot3 (not renamed but obs are organized in 4 consecutive images)
```text
├── BENI_TUB_S3_2V_F3_UNK_0106.jpg
├── BENI_TUB_S3_2V_F3_UNK_0107.jpg
├── BENI_TUB_S3_2V_F3_UNK_0108.jpg
├── BENI_TUB_S3_2V_F3_UNK_0109.jpg
├── BENI_TUB_S3_2V_F3_UNK_0110.jpg
├── BENI_TUB_S3_2V_F3_UNK_0111.jpg
├── BENI_TUB_S3_2V_F3_UNK_0112.jpg
├── BENI_TUB_S3_2V_F3_UNK_0113.jpg
...
```

## Notes on data

Data are organized into 11 sets corresponding to acquisitions campaign defined in the project roadmap.
Those sets are sometimes split into parts to maintain naming convention within each part as shown below :

```text
data/raw
├── lot1-20-04-2023-benitiers
├── lot1-20-04-2023-sediments
├── lot2-30-05-2023-tak_nacl
├── lot2-30-05-2023-tak_nai
├── lot2-30-05-2023-tak_nai-part2
├── lot3-08-06-2023-benitiers
├── lot4-28-06-2023-sediments-part1
├── lot4-28-06-2023-sediments-part2
├── lot4-28-06-2023-sediments-part3
├── lot5-04-07-2023-benitiers-part1
├── lot5-04-07-2023-benitiers-part2
├── lot6-12-08-2023-eau-horizontal
├── lot6-12-08-2023-eau-vertical
├── lot7-28-09-2023-benitiers
├── lot8-28-09-2023-benitiers
├── lot9-09-10-2023-benitiers
├── lot10-09-10-2023-benitiers 
└── lot11-20-11-2023-eau            
```

## Pipeline

### 0. Configuration

Refer to `configs/default_config.yaml` to reproduce all the results. 
This config file defines : 
- data folders to ingest and process data
- modelling experiments directory
- fiftyone evaluations directory
- csv export directory
- parameters to run the different pipeline components

For unit testting the pipeline, a specific `configs\test_config.yaml` is defined.

An example is also provided to run the pipeline for `lot1` under `configs/config_lot1.yaml`


### 1. Data preparation

#### Step 1 : compute embeddings centers for each filters on dataset lot2

Use `data_prep/compute_embeddings_filter_centers_lot2.ipynb` in order to save embeddings centers.
We use `lot2` instead of `lot1` because it is more challenging with some overlap between TRI and NAT


#### Step 2 : ingest data 

- find metadata to rename files : use OCR to read zoom and use exif to read exposition
- filter to maintain 4 images per acquisition : this is required to infer the type of filter used as this information 
is not always correct due to manual renaming. In order to know which images belong to the same observation, multiple cases
are handled. Refer to the data naming convention section [here](#Data-naming-convention).
- infer color based on 4 images using embeddings centers matching
- filter valid acquisitions : keep only zoom = 500 200 with image size = 1920x1200

```bash
# add current dir to pythonpath
export PYTHONPATH=$PWD
# to process specific data set
python src/pipeline.py ingest_data_subset configs/default_config.yaml lot1-20-04-2023-benitiers

# to process all the data in default_config.DATA.RAW
python src/pipeline.py ingest_data configs/default_config.yaml
```

#### Step 3 : create and export composite dataset

```bash
# add current dir to pythonpath
export PYTHONPATH=$PWD
# to process specific data set
python src/pipeline.py create_composite_subset configs/default_config.yaml lot1-20-04-2023-benitiers

# to process all the data in default_config.DATA.RAW
python src/pipeline.py create_composite configs/default_config.yaml
```


### 2. labkit_labeling (contains manual steps)

To label data, we grouped images into smaller annotation tasks to limit the variability within group. This allows us to 
learn a simple pixel classifier model efficiently using labkit interactive learning. Also, we limit the number of samples 
within groups to avoid labkit memory limitation when dealing with large image sets. Because grouping images into 
annotation tasks change the image folder structure, we need to also get back the original folder structure once masks
are computed to pair masks with images from fiftyone dataset which contains all the metadata. 

#### Step 4 : create annotation tasks (manual - varies with datasets lots)  

At first, we used all the data and made clusters to facilitate labkit annotations 
using coherent batches so that one Labkit model is able to deal with the whole batch

For lot4, we sample the data based on its origin (sample_type + island + station + replicas). 
Refer to `viz_datasets_composite_v2.ipynb`.

To sample data from lot 5 to 10, refer to `viz_datasets_composite_and_sample_data_for_annotation.ipynb`
We cluster the data into 30 clusters then sample 25% and take 20 samples maximum from each cluster.

```bash
# add current dir to pythonpath
export PYTHONPATH=$PWD
# to reproduce lot1 to lot 3 annotation task creation
python src/pipeline.py create_tasks configs/default_config.yaml lot1_3
# to reproduce lot4 annotation task creation
python src/pipeline.py create_tasks configs/default_config.yaml lot4
# to reproduce lot5 to lot 10 annotation task creation
python src/pipeline.py create_tasks configs/default_config.yaml lot5_10

# to create tasks with subset called new_lot, edit default_config CREATE_TASKS section for create_tasks params
python src/pipeline.py create_tasks configs/default_config.yaml new_lot
```

#### Step 5 : labkit annotation + labkit inference  

Follow this tutorial : https://docs.google.com/presentation/d/12bUywRMCjIyrB3BmrCNps7Y_XApCsKtEKgkffKQYOjs/edit#slide=id.p

A model is saved for each annotation task with the same name as the task folder name under `data/processed/labkit_models`

It is then possible to perform inference using labkit script to obtain the masks for the whole folder. 
Follow the imageJ macro scripts to generate segmentation masks under `data/processed/annotated_data` :
```
src/labkit_labeling/labkitmacro_resize_lot3.ijm
src/labkit_labeling/labkitmacro_resize_lot4.ijm
src/labkit_labeling/labkitmacro_resize_lot5_10.ijm
src/labkit_labeling/labkitmacro_resize_lot1_lot4_review_beni.ijm
src/labkit_labeling/labkitmacro_resize_lot1_lot4_review_sed.ijm
```
*Note : Unfortunately, we cannot reproduce the first annotation of lot1 and lot2 with labkit.  
We only saved the final masks for lot1 and lot2 in `labkitinference` folder.
Those are refined using [re-annotation section](#reannotation)

Then, reorganize masks into the original data structure to pair image and mask correctly using the script below.
```bash
# add current dir to pythonpath
export PYTHONPATH=$PWD

python src/pipeline.py matching_old_names_with_new configs/default_config.yaml lot3
python src/pipeline.py matching_old_names_with_new configs/default_config.yaml lot4
python src/pipeline.py matching_old_names_with_new configs/default_config.yaml lot5_10
python src/pipeline.py matching_old_names_with_new configs/default_config.yaml lot1_lot4_review_beni
python src/pipeline.py matching_old_names_with_new configs/default_config.yaml lot1_lot4_review_sed

# for new lot named new_lot
python src/pipeline.py matching_old_names_with_new configs/default_config.yaml new_lot
```

#### Step 6 : generate annotated dataset  

Refer to the script `generate_annotated_dataset.py`
```bash
export PYTHONPATH=$PWD
# to reproduce on subset
python src/pipeline.py generate_annotated_subset configs/default_config.yaml lot1-20-04-2023-benitiers
# to reproduce on all datasets
python src/pipeline.py generate_annotated_dataset configs/default_config.yaml
```

Samples are tagged as train, test or unlabelled. 30% of data is used and data is split by origins.

#### Step 7 : export fiftyone dataset to Image Sequence format and save train/test protocols

Protocols are defined as follows :
```bash
export PYTHONPATH=$PWD

# to convert to image sequence dataset format for all datasets
python src/pipeline.py prepare_dataset_for_openmmseg configs/default_config.yaml
```

### 3. Modeling

Use `openmmseg` framework to train, evaluate and predict segmentation masks for microplastic detection.

**Modeling Workflow Overview:**
1. **Train models** using mmsegmentation with custom dataset and transforms
2. **Run inference** on test/unlabeled data to generate prediction masks
3. **Evaluate performance** using the new modular evaluation system:
   - Semantic segmentation metrics (IoU, Dice, precision, recall)
   - Instance segmentation metrics (mAP, per-detection precision/recall)
   - Multi-model comparison on the same FiftyOne dataset

**Evaluation System:**
- **New (Recommended):** `src/modeling/evaluate_segmentation.py` - Modular CLI supporting both semantic and instance evaluation
- **Legacy:** `src/modeling/run_fiftyone_eval.py` - Original evaluation script (still supported)

Our contributions to mmseg :

#### Custom Dataset Class
**File:** `mmseg/datasets/microplastic.py` (61 lines)
- **Registered as:** `MicroPlasticDataset`
- **Purpose:** Custom dataset that reads from text protocol files (train/test splits)
- **Key features:**
  - 2 classes: 'background' and 'microplastic'
  - Reads image paths from annotation files (e.g., `train_EvalProtocol_TRAIN_TEST.txt`)
  - Supports `.jpg` images and `.png` masks
  - No zero-label reduction (keeps background as class 0)
- **Registration:** Added to `mmseg/datasets/__init__.py:27` and exported in `__all__:63`

#### Custom Transforms
**File:** `mmseg/datasets/transforms/custom_transforms.py` (160 lines)

**A. `InvertBinaryLabels`**
- **Purpose:** Converts binary masks from 255→1 and applies Gaussian blur
- **Why:** Labkit outputs masks with 255 for microplastics; this normalizes to 1 and smooths edges
- **Registration:** Added to `mmseg/datasets/transforms/__init__.py:17` and exported in `__all__:29`

**B. `RandomCropForeground`**
- **Purpose:** Intelligent cropping that focuses on microplastic regions
- **Algorithm:**
  1. Randomly selects a foreground pixel (class 1)
  2. Crops with 25-75% overlap to ensure microplastic in frame
  3. Falls back to random crop if no foreground exists
- **Why:** Addresses severe class imbalance (microplastics are tiny compared to background)
- **Parameters:**
  - `crop_size`: (400, 400) or (256, 256)
  - `cat_max_ratio`: Controls class distribution in crop
  - `ignore_index`: 255 (default)
- **Note:** Imported via config's `custom_imports` directive

#### Custom Inference Script
**File:** `tools/inference.py` (73 lines)
- **Purpose:** Run inference on unlabeled images with custom post-processing
- **Key differences from standard mmseg:**
  - Applies **sigmoid** to logits instead of argmax
  - Uses **threshold=0.5** for binary segmentation
  - Saves as binary PNG masks (0 or 255)
  - Processes entire directories of `.jpg` images

#### Project Configs
**Directory:** `projects/microplastic_detection/configs/`

**Base Configs:**
- `microplastic_detection_256x256.py` - 256×256 crop size pipeline
- `microplastic_detection_400x400.py` - 400×400 crop size pipeline

**Key configuration:**
```python
dataset_type = 'MicroPlasticDataset'
data_root = 'data/processed/prepare_dataset_for_openmmseg'
custom_imports = dict(imports='mmseg.datasets.transforms.custom_transforms')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='InvertBinaryLabels'),  # Custom transform
    dict(type='RandomResize', scale=(625, 1000), ratio_range=(0.8, 1.2)),
    dict(type='RandomCropForeground', crop_size=crop_size),  # Custom transform
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', ...),
    dict(type='PackSegInputs')
]
```

**Training Configs (inherit from base configs):**
1. `fcn-unet-s5-d16_...-256x256_train_test.py` - 256px, TRAIN_TEST protocol
2. `fcn-unet-s5-d16_...-256x256_sed_intra_inter_ile.py` - 256px, sediment protocol
3. `fcn-unet-s5-d16_...-400x400_train_test.py` - 400px, TRAIN_TEST protocol
4. `fcn-unet-s5-d16_...-400x400_beni_hao_mak_tub.py` - 400px, BENI_HAO_MAK_TUB protocol

**Model configuration:**
- Architecture: FCN-UNet (S5-D16)
- Loss: Dice Loss with sigmoid
- Optimizer: SGD (lr=0.01, momentum=0.9)
- Scheduler: PolyLR (eta_min=1e-4)
- Training: 8000 iterations, validate every 200
- Test mode: Sliding window (crop_size with stride=300)

#### Evaluation Protocols
**File:** `src/labkit_labeling/prepare_dataset_for_openmmseg.py`

Available protocols for train/test splits:
1. **TRAIN_TEST** (5): Standard train/test tags from FiftyOne (30% test)
2. **BENI_INTRA_INTER_ILE** (1): Benitiers only, train on TAK island, test on all islands
3. **BENI_INTRA_ILE** (2): Benitiers only, standard train/test split
4. **SED_INTRA_INTER_ILE** (3): Sediments only, train on lot1, test on all sediments
5. **SED_BENI_INTRA_INTER_ILE** (4): Combined sediment+benitier training
6. **UNLABELLED** (6): Unlabeled samples only
7. **BENI_HAO_MAK_TUB** (7): Benitiers from HAO, MAKEMO, TUBUAI islands

Protocol files are generated as `.txt` files containing relative image paths, referenced in training configs via the `ann_file` parameter.

TODO create a pull request to add this project into mmseg public repo

#### Step 8.1 : train and evaluate

Train and eval using `mmsegmentation`
```bash
pixi shell
export PYTHONPATH=mmsegmentation:$PWD

# to reproduce exp with beni_3_islands protocol with 400*400 input size
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_beni_hao_mak_tub.py 

# to reproduce exp with sed_inta_inter_ile protocol with 256*256 input size
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_sed_intra_inter_ile.py

# to reproduce exp with train_test protocol with 256*256 input size
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test.py

# to reproduce exp with train_test protocol with 400*400 input size
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py

python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_beni_hao_mak_tub.py 

```

#### Step 8.2 : inference to use as input for fiftyone eval

```bash
pixi shell
export PYTHONPATH=mmsegmentation:$PWD

# example of using model for inference on unlabelled data
python mmsegmentation/tools/inference.py \
--model_config mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py \
--model_ckpts work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/best_mIoU_iter_2800.pth \
--img_folder data/processed/create_composite/lot11-20-11-2023-eau/data \
--save_folder work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/inference/lot11-20-11-2023-eau

# reproduce article beni
./scripts/run_inference_article_beni.sh
```

#### Step 8.3 : evaluate models (semantic and instance segmentation)

**Granular Evaluation Workflow (Recommended)**

Use the modular evaluation scripts for step-by-step semantic and instance segmentation evaluation:

```bash
pixi shell
export PYTHONPATH=$PWD

# Example for article beni evaluations
config=fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_beni_hao_mak_tub
dataset=mp_article_beni

# Step 1: Add predictions to dataset
python src/modeling/add_predictions.py \
  --dataset_name $dataset \
  --predictions_dir work_dirs/$config/inference \
  --config_name $config.py \
  --filter_tags test

# Step 2: Evaluate semantic segmentation (IoU, Dice, precision, recall)
python src/modeling/evaluate_sem_seg.py \
  --dataset_name $dataset \
  --pred_field predictions_400x400_beni_hao_mak_tub \
  --filter_tags test

# Step 3: Convert ground truth to instances (ONCE per dataset)
python src/modeling/convert_to_instance_segmentation.py \
  --dataset_name $dataset \
  --mask_field ground_truth \
  --det_field inst_ground_truth \
  --filter_tags test

# Step 4: Convert predictions to instances
python src/modeling/convert_to_instance_segmentation.py \
  --dataset_name $dataset \
  --mask_field predictions_400x400_beni_hao_mak_tub \
  --det_field inst_predictions_400x400_beni_hao_mak_tub \
  --filter_tags test

# Step 5: Evaluate instance segmentation (mAP, per-instance metrics)
python src/modeling/evaluate_inst_seg.py \
  --dataset_name $dataset \
  --pred_field inst_predictions_400x400_beni_hao_mak_tub \
  --gt_field inst_ground_truth \
  --filter_tags test
```

**Key Features:**
- Granular control: Run individual steps for easier debugging
- Config-based field naming prevents conflicts when evaluating multiple models
- Supports both semantic and instance segmentation evaluation
- Uses existing `convert_to_instance_segmentation.py` for mask-to-instance conversion
- Conversion script remains standalone for future instance segmentation training workflows

See `docs/EVALUATION_WORKFLOW.md` for detailed documentation and examples.

**Legacy Workflow (Old Method)**

For backward compatibility, the old evaluation script is still available:
```bash
pixi shell
export PYTHONPATH=$PWD

# Create dataset and run evaluation
python src/modeling/run_fiftyone_eval.py \
  data/processed/generate_annotated_dataset \
  data/processed/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference \
  --eval_bool True
```

#### Step 8.4 : compare multiple models

The evaluation system allows you to evaluate multiple models on the same FiftyOne dataset without field conflicts:

```bash
pixi shell
export PYTHONPATH=$PWD

# Evaluate Model 1: 400x400 train_test
config1=fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test

python src/modeling/add_predictions.py \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/$config1/inference \
  --config_name $config1.py

python src/modeling/evaluate_sem_seg.py \
  --dataset_name mp_dataset \
  --pred_field predictions_400x400_train_test \
  --filter_tags test

python src/modeling/convert_to_instance_segmentation.py convert_dataset \
  --dataset_name mp_dataset \
  --mask_field predictions_400x400_train_test \
  --det_field inst_predictions_400x400_train_test

python src/modeling/evaluate_inst_seg.py eval_instances \
  --dataset_name mp_dataset \
  --pred_field inst_predictions_400x400_train_test \
  --gt_field inst_ground_truth \
  --filter_tags test

# Evaluate Model 2: 256x256 sed_intra_inter_ile
config2=fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_sed_intra_inter_ile

python src/modeling/add_predictions.py \
  --dataset_name mp_dataset \
  --predictions_dir data/modeling/work_dirs/$config2/inference \
  --config_name $config2.py

python src/modeling/evaluate_sem_seg.py \
  --dataset_name mp_dataset \
  --pred_field predictions_256x256_sed_intra_inter_ile \
  --filter_tags test

python src/modeling/convert_to_instance_segmentation.py convert_dataset \
  --dataset_name mp_dataset \
  --mask_field predictions_256x256_sed_intra_inter_ile \
  --det_field inst_predictions_256x256_sed_intra_inter_ile

python src/modeling/evaluate_inst_seg.py eval_instances \
  --dataset_name mp_dataset \
  --pred_field inst_predictions_256x256_sed_intra_inter_ile \
  --gt_field inst_ground_truth \
  --filter_tags test

# Now mp_dataset contains:
# - predictions_400x400_train_test & predictions_256x256_sed_intra_inter_ile
# - inst_predictions_400x400_train_test & inst_predictions_256x256_sed_intra_inter_ile
# - eval_400x400_train_test & eval_256x256_sed_intra_inter_ile
# - eval_inst_400x400_train_test & eval_inst_256x256_sed_intra_inter_ile
```

Compare results in FiftyOne App:
```python
import fiftyone as fo

dataset = fo.load_dataset("mp_dataset")
session = fo.launch_app(dataset.match_tags("test"))
# Toggle between prediction fields in the UI to compare models
```

#### Step 8.5 (optional) : save dataset to disk

```python
import fiftyone as fo

dataset = fo.load_dataset("mp_dataset")
dataset.export(
    export_dir="data/processed/fiftyone_evaluations/ds_export",
    dataset_type=fo.types.FiftyOneDataset,
)
```

#### Step 8.6 (optional) : advanced instance conversion

The new evaluation workflow automatically converts semantic masks to instance segmentation (see Step 8.3).
However, you can also manually convert masks using the standalone conversion script for custom use cases:

**Using the Python API:**
```python
import fiftyone as fo
from src.modeling.convert_to_instance_segmentation import add_instance_segmentation_to_dataset

# Load your dataset
dataset = fo.load_dataset("mp_dataset")

# Add instance detections to a new field
add_instance_segmentation_to_dataset(
    dataset,
    mask_field="ground_truth",      # Source mask field to convert
    det_field="detections",         # Target detection field to create
    compute_scores=True,            # Compute contrast scores and RGB values
    min_area=40,                    # Filter out detections < 40 pixels
    max_area=160000,                # Filter out detections > 160000 pixels
    batch_size=100                  # Save every N samples
)
```

**Using the CLI:**
```bash
pixi shell
export PYTHONPATH=$PWD

# Convert ground truth masks to detections
python src/modeling/convert_to_instance_segmentation.py convert_dataset \
    --dataset_name mp_dataset \
    --mask_field ground_truth \
    --det_field detections \
    --compute_scores True \
    --min_area 40 \
    --max_area 160000
```

**Detection Attributes:**
Each instance includes MP-VAT shape descriptors:
- `area`, `perimeter`
- `feret_diameter`, `feret_degree`
- `circularity`, `roundness`
- `mp_shape`: Classification (Fibers, Fragments, Particles)
- `score`: Contrast-based quality metric
- `mean_red`, `mean_green`, `mean_blue`: RGB values

**Note:** The new `evaluate_segmentation.py` script (Step 8.3) handles this conversion automatically with the `inst_` prefix convention.

### 4. Export CSVs

In order to export CSV with instance segmentation results and descriptors similar to MP-VAT2.0, we need to convert 
masks (predictions or ground truth) into instance detections or polylines. Shape descriptors and custom confidence score 
is computed for each detection. 

When someone adds a new lot of samples, make sure the following re-requisite are followed before exporting :
- create image composite dataset as described in [section 3](#step-3--create-and-export-composite-dataset)
- make sure mask predictions are inferred as described in [section 8.2](#Step-8.2)

To export detections for a new lot in CSV format run :
```bash
pixi shell
export PYTHONPATH=$PWD

# export a specific unlabelled folder (ex new unlabelled lots with lot11-20-11-2023-eau)
python src/pipeline.py export_unlabelled_folder configs/default_config.yaml \
data/processed/create_composite/lot11-20-11-2023-eau \
data/processed/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference/lot11-20-11-2023-eau
```

To export all the lots, make sure that :
- all lots are either annotated manually (labkit) or have predictions or both
- then run the following command
```bash
pixi shell
export PYTHONPATH=$PWD

# export annotated dataset
python src/pipeline.py export configs/default_config.yaml data/processed/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference
```
When both masks (labkit and predictions) are available, exports use the labkit gt annotations.


## Visualize datasets

Use the following notebooks :
- `src/viz/viz_lot5_10_and_sample_data_for_annotation.ipynb` 
- `src/viz/viz_lot1_4_datasets_composite_v2.ipynb` 

Or directly go to remote_fiftyone project

TODO include remote fiftyone as submodule

## Reannotation

Either model predictions or manual annotations can be of low quality and must be corrected.
Here is the procedure to complete the correction :
- Use fiftyone to visualize gt masks and tag samples that need re-annotation  
- Use notebook `load_lot1_lot4_tags_for_reannotation.ipynb` to create reannotation tasks for beni and sed  
- Get annotations and copy them to `/home/taiamiti/Projects/micro-plastic/cmdinferencelabkit/`  
- Perform inference using labkit model with `labkitmacro_resize_lot1_lot4_review_beni.ijm` and 
`labkitmacro_resize_lot1_lot4_review_sed.ijm`
- Copy masks to `data/processed/labelkitinference` using `src/labkit_labeling/matching_old_names_with_new.py`
- Generate annotated datasets using `src/labkit_labeling/generate_annotated_dataset.py`


## Detection filtering

Manual annotations using Labkit and predictions using `openmmseg` both generate some detections
that can be noisy and additional filtering is necessary based on a combination of criteria.
Use fiftyone to inspect the detections for each lot and tune the score parameter to filter the valid 
detections on the given image set. 



