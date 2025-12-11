import os
import fiftyone as fo
from fiftyone import ViewField as F

from src.labkit_labeling.prepare_dataset_for_openmmseg import load_dataset_from_multiple_sources


ds_root = "data/processed/generate_annotated_dataset/"
ds_paths = [os.path.join(ds_root, ds_path) for ds_path in os.listdir(ds_root)]
dataset = load_dataset_from_multiple_sources(ds_paths)
dataset.name = "mp_dataset"
dataset.persistent = True
dataset.save()
print(f'Dataset loaded: {dataset.name}')
print(f'Samples: {len(dataset)}')


dataset_article_beni = (fo.load_dataset("mp_dataset")
                          .match(F('island').is_in(("MAK", "TUB", "HAO")))
                          .match(F('sample_type').is_in(("BENI", "CBENI")))).clone("mp_article_beni")
dataset_article_beni.persistent = True
dataset.save()