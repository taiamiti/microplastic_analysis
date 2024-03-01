import os

import fiftyone as fo
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.labkit_labeling.generate_annotated_dataset import segment_sample, score_sample

"""
Export CSVs from annotated datasets :
- load annotated datasets
- add predictions on unlabelled
- recompute score on detections and polylines
- export using manual annotations (labkit predictions) if exist otherwise use model predictions
"""


def dataset_as_dataframe(dataset):
    dataset_dict = dataset.to_dict()
    dfs = []
    for sample in tqdm(dataset_dict["samples"]):
        dets = sample["detections"]['detections']
        for det in dets:
            for key in ["_cls", "tags", "_id", "mask"]:
                del det[key]
        sample["detections"] = dets
        sample_fields = ['filepath', 'image_path', 'sample_type', 'island', 'station', 'replica', 'distil',
                         'sample_id', 'image_id', 'filter', 'extra', 'exposure_time', 'zoom', 'obs_id']
        dfi = pd.json_normalize(sample, "detections")
        for sample_field in sample_fields:
            dfi[sample_field] = sample[sample_field]
        if len(sample["tags"]) >= 2:
            dfi["subset"] = sample["tags"][0]
            dfi["lot"] = sample["tags"][1]
        else:
            dfi["lot"] = sample["tags"][0]
        dfi["image_width"] = sample["metadata"]["width"]
        dfi["image_height"] = sample["metadata"]["height"]
        dfs.append(dfi)
    if len(dfs) > 0:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()
    return df


def export_dataset_as_csv(dataset, csv_export_path):
    # Export CSV
    df = dataset_as_dataframe(dataset)  # .drop(["_id", "_cls", "mask"], axis=1)
    os.makedirs(os.path.dirname(csv_export_path), exist_ok=True)
    df.to_csv(csv_export_path, index=False)


def create_dataset(data_root_dir):
    dataset_dirs = os.listdir(data_root_dir)
    export_dirs = [os.path.join(data_root_dir, dataset_dir) for dataset_dir in dataset_dirs]
    datasets_names = [dataset_dir.strip("-part1").strip("-part2").strip("-part3") for dataset_dir in dataset_dirs]
    # Merge to single dataset
    dataset = fo.Dataset()
    for export_dir, dataset_name in zip(export_dirs, datasets_names):
        dataset.add_dir(
            dataset_dir=export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            tags=dataset_name,
        )
    return dataset, datasets_names


def add_prediction_on_unlabelled(dataset, labels_path, label_ext=".png"):
    with dataset.save_context() as context:
        for sample in tqdm(dataset.match_tags("unlabelled")):
            p = Path(sample.filepath)
            rel_path = str(p.relative_to(p.parents[2])).replace(".jpg", label_ext)
            mask_path = os.path.abspath(os.path.join(labels_path, rel_path))
            if os.path.exists(mask_path):
                sample["prediction"] = fo.Segmentation(mask_path=mask_path)
                sample['detections'], sample['polylines'] = segment_sample(sample["prediction"]["mask_path"])
                score_sample(sample, mask_key="prediction", det_key='detections', poly_key="polylines")
                context.save(sample)
    dataset.add_dynamic_sample_fields()
    dataset.save()


def add_score_on_gt_samples(dataset):
    with dataset.save_context() as context:
        for sample in tqdm(dataset.match_tags(["train", "test"])):
            sample['detections'], sample['polylines'] = segment_sample(sample["ground_truth"]["mask_path"])
            score_sample(sample, mask_key="ground_truth", det_key='detections', poly_key="polylines")
            context.save(sample)
    dataset.add_dynamic_sample_fields()
    dataset.save()


def main_export(data_root_dir, labels_root_dir, export_dir):
    """Export annotated dataset (can contain unlabelled data) from generate_annotated_dataset

    Args:
        data_root_dir:
        labels_root_dir:
        export_dir:

    Returns:

    """
    ds, datasets_names = create_dataset(data_root_dir)
    add_prediction_on_unlabelled(ds, labels_root_dir)
    add_score_on_gt_samples(ds)
    dataset_dirs = os.listdir(data_root_dir)
    datasets_names = [dataset_dir.strip("-part1").strip("-part2").strip("-part3") for dataset_dir in dataset_dirs]
    for datasets_name in datasets_names:
        view = ds.match_tags(datasets_name)
        export_dataset_as_csv(view, os.path.join(export_dir, datasets_name + ".csv"))


def main_export_unlabelled_folder(dataset_path, labels_path, export_dir):
    """Export unlabelled folder when new data are acquired

    Args:
        export_dir:
        dataset_path:
        labels_path:

    Returns:

    """
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_path,
        dataset_type=fo.types.FiftyOneDataset,
        tags=dataset_name
    )
    dataset.tag_samples("unlabelled")
    add_prediction_on_unlabelled(dataset, labels_path)
    dataset.add_dynamic_sample_fields()
    dataset.save()
    export_dataset_as_csv(dataset, os.path.join(export_dir, dataset_name + ".csv"))


if __name__ == '__main__':
    # data_root_dir = "data/processed/generate_annotated_dataset"
    # labels_root_dir = "data/processed/mmsegmentation/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/pred_unlabelled"
    # labels_root_dir = "data/processed/mmsegmentation/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/pred"
    # export_dir = "data/processed/exporter"
    # main_export(data_root_dir, labels_root_dir)

    main_export_unlabelled_folder(
        dataset_path="data/processed/create_composite/lot11-20-11-2023-eau",
        labels_path="work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/pred_unlabelled",
        export_dir="data/processed/exporter"
    )

