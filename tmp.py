import fiftyone as fo

dataset = fo.Dataset.from_dir(
  dataset_dir='/home/taiamiti/Projects/microplastic_analysis/data/processed/fiftyone_evaluations/ds_export',
  dataset_type=fo.types.FiftyOneDataset,
  name='mp_dataset'
)
dataset.persistent = True
dataset.save()
print(f'Dataset loaded: {dataset.name}')
print(f'Samples: {len(dataset)}')

