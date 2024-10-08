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

The code is structured into 4 parts :
- data_prep : scripts to prepare data and create image composite
- labkit_labeling : scripts to analyse, cluster and annotate data
- modeling : train and eval scripts
- export : export dataset (fiftyone) with ground truths or predictions to csv

The whole pipeline is depicted in the figure below :
todo

## Installation

There are 2 environments to run the project : one for data engineering (data_prep, labkit_labeling, export) 
and one for modeling. This is because modeling is based on mmsegmentation framework which has an independent installation 
procedure which is quite heavy and can conflict with torch installation in data engineering for computing clip embeddings.
For this reason the two envs are kept separated.

### 1. data engineering env

Use conda to create an environment for data engineering tasks
```bash
conda create -n map_de python=3.9
conda activate map_de
pip install -r requirements_de.txt
```

### 2. modeling env

Follow the installation instruction of mmsegmentation for the modeling environment 
(cf [README.md](./mmsegmentation/README.md))

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
Follow their installation instruction to create a conda environment.

Our contributions to mmseg :
- add custom transform `InvertBinaryLabels` and `RandomCropForeground`
- add microplastic dataset `MicroPlasticDataset`
- add inference script `tools/inference.py`
- fix image demo script
- add configs for microplastic detection training and eval

TODO create a pull request to add this project into mmseg public repo

#### Step 8.1 : train and evaluate  

Train and eval using `mmsegmentation` 
```bash
export PYTHONPATH=mmsegmentation:$PWD
conda activate openmmlab

# to reproduce exp with sed_inta_inter_ile protocol with 256*256 input size
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_sed_intra_inter_ile.py \
--work-dir data/modeling/work_dirs

# to reproduce exp with train_test protocol with 256*256 input size
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test.py \
--work-dir data/modeling/work_dirs

# to reproduce exp with train_test protocol with 400*400 input size
python mmsegmentation/tools/train.py \
mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py \
--work-dir data/modeling/work_dirs
```

#### Step 8.2 : inference to use as input for fiftyone eval

```bash
conda activate openmmlab

# example of using model for inference on unlabelled data
python mmsegmentation/tools/inference.py \
--model_config mmsegmentation/projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py \
--model_ckpts work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/best_mIoU_iter_2800.pth \
--img_folder data/processed/create_composite/lot11-20-11-2023-eau/data \
--save_folder work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/inference/lot11-20-11-2023-eau
```

#### Step 8.3 : visualize evaluations  

To run evaluations and visualize results in fiftyone UI, run the following :
```bash
conda activate map_de

# example of using model for inference on unlabelled data
python src/modeling/run_fiftyone_eval.py \
data/processed/generate_annotated_dataset \
data/processed/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference \
--eval_bool True
```

#### Step 8.4 (optional) : save dataset to disk

```python
import fiftyone as fo

dataset = fo.load_dataset("mp_dataset")
dataset.export(
    export_dir="data/processed/fiftyone_evaluations/ds_export",
    dataset_type=fo.types.FiftyOneDataset,
)
```

### 4. Export CSVs

In order to export CSV with instance segmentation results and descriptors similar to MP-VAT2.0, we need to convert 
masks (predictions or ground truth) into instance detections or polylines. Shape descriptors and custom confidence score 
is computed for each detection. 

When someone adds a new lot of samples, make sure the following re-requisite are followed before exporting :
- create image composite dataset as described in [section 3](#step-3--create-and-export-composite-dataset)
- make sure mask predictions are inferred as described in [section 8.2](#Step-8.2)

To export detections for a new lot in CSV format run :
```bash
conda activate map_de
# export a specific unlabelled folder (ex new unlabelled lots with lot11-20-11-2023-eau)
python src/pipeline.py export_unlabelled_folder configs/default_config.yaml \
data/processed/create_composite/lot11-20-11-2023-eau \
data/processed/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference/lot11-20-11-2023-eau
```

To export all the lots, make sure that :
- all lots are either annotated manually (labkit) or have predictions or both
- then run the following command 
```bash
conda activate map_de

# export annotated dataset
python src/pipeline.py export configs/default_config.yaml data/processed/work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-256x256_train_test/inference
```
When both masks (labkit and predictions) are available, exports use the labkit gt annotations.


## Visualize datasets

`viz_datasets.ipynb` : view raw images datasets
`viz_datasets_composite_and_sample_data_for_annotation.ipynb` : view composite dataset   
`viz_datasets_composite_v2.ipynb` : ... 

Or directly go to remote_fiftyone project

TODO include remote fiftyone as submodule

## Reannotation

Either model predictions or manual annotations can be of low quality and must be corrected.
Here is the procedure to complete the correction :
- Use fiftyone to visualize gt masks and tag samples that need re-annotation  
- Use notebook `load_lot1_lot4_tags_for_reannotation.ipynb` to create reannotation tasks for beni and sed  
- Get annotations and copy them to `/home/taiamiti/Projects/micro-plastic/cmdinferencelabkit/`  
- Perform inference using labkit model with `labkitmacro_resize_reannot_lot1-4_beni.ijm` and `labkitmacro_resize_reannot_lot1-4_sed.ijm`  
- Copy masks to `data/processed/labelkitinference` using `src/matching_old_names_with_new.py`
- Generate annotated datasets using `src/generate_annotated_dataset.py`


## Detection filtering

Manual annotations using Labkit and predictions using `openmmseg` both generate some detections
that can be noisy and additional filtering is necessary based on a combination of criteria.
Use fiftyone to inspect the detections for each lot and tune the score parameter to filter the valid 
detections on the given image set. 



