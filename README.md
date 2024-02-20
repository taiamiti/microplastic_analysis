~~# My Notes

## Install

Use conda to create a python=3.9 env
```bash
conda create -n map python=3.9
conda activate map
pip install -r requirements.txt
```

## Data naming convention

3 cases are handled :
```python
from enum import IntEnum

class DataPrior(IntEnum):  # naming convention todo rename this
    RENAMED = 1  # ex : lot 1
    PARTIAL_RENAMED = 2  # ex lot2 (only obs id renamed ie: CSED_TAK_S1_UNK_UNK_0001 (1).jpg)
    CONSECUTIVE = 3  # ex lot3 (not renamed but obs are organized in 4 consecutive images
    # ie: BENI_MAK_S1_5V_F2_UNK_0000.jpg)
```
RENAMED : all files are renamed as follows `{sample_type}_{island}_{station}_{replica}_{distil}_{sample_id}`

## Pipeline

Step 1 : compute embeddings centers for each filters on dataset lot1 and lot2
`compute_embeddings_filter_centers_lot1.ipynb`
`compute_embeddings_filter_centers_lot2.ipynb`

Step 2 : ingest data using script `ingest_data.py`
- find metadata to rename files (zoom, exposition, )
- filter to maintain 4 images per acquisition (prior to finding filter)
- infer color based on 4 images using embeddings centers matching
- keep only zoom = 500 200 + image size = 1920x1200 + new_filter = dapi tri  


Step 3 : create and export composite dataset using `create_composite.py`  
Note : clearml was first used to handle these but was removed due to complications
instead use pipeline.py to directly call the functions


Step 4 : create annotation tasks (manual - varies with datasets lots)  
At first, we used all the data and made clusters to facilitate labkit annotations 
using coherent batches so that one Labkit model is able to deal with the whole batch

Refer to `create_tasks_lotx.ipynb`.  
Refer to `viz_datasets_composite_v2.ipynb` for lot 4.  
To sample data from lot 5 to 10, refer to `viz_datasets_composite_and_sample_data_for_annotation.ipynb`
We cluster the data into 15 clusters and take 20 samples from each cluster.


Step 5.1 : labkit annotation + labkit inference  
Use a model per image folder (clusters). Follow the imageJ macro scripts to generate segmentation masks :
```
cmdinferencelabkit/labkitmacro_resize_reannot_lot1-4_beni.ijm
cmdinferencelabkit/labkitmacro_resize_reannot_lot1-4_sed.ijm
cmdinferencelabkit/labkitmacro_resize_reannot_lot5-10.ijm
```

Step 5.2 : model inference  
Use the mmsegmentation inference script to generate segmentation masks :
```
mmsegmentation/tools/inference.py
```

Step 6 : generate annotated dataset  
Refer to the script `generate_annotated_dataset.py`

Step 7 : export CSVs  
Refer to the script `exporter.py`


## Visualize datasets

`viz_datasets.ipynb` : view raw images datasets
`viz_datasets_composite_and_sample_data_for_annotation.ipynb` : view composite dataset   
`viz_datasets_composite_v2.ipynb` : ... 

Or directly go to remote_fiftyone project


## Notes data

```text
.
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
└── lot11-20-11-2023-eau            # unannotated
```

## Modelling

Use `openmmseg` framework to train, evaluate and predict segmentation masks for microplastic detection.
Once annotated datasets are ready (generate annotated datasets using `src/generate_annotated_dataset.py`)
do the following steps :
- Use `notebook/prepare_dataset_for_openmmseg.ipynb` to prepare dataset for openmmseg
- Train eval using `mmsegmentation`
- viz using fiftyone (use inference to generate masks then use remote_fiftyone main script to load and evaluate with fiftyone)
refer to `fiftyone_evaluations.ipynb`


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



