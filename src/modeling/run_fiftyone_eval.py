import fiftyone as fo
import os
from pathlib import Path

import fire


def create_persistent_dataset(dataset_root_dir, dataset_name="fo_eval_dataset"):
    dataset_dirs = os.listdir(dataset_root_dir)
    export_dirs = [os.path.join(dataset_root_dir, dataset_dir) for dataset_dir in dataset_dirs]
    datasets_names = [dataset_dir.strip("-part1").strip("-part2").strip("-part3") for dataset_dir in dataset_dirs]

    # Merge to single dataset
    dataset = fo.Dataset(name=dataset_name, persistent=True, overwrite=True)
    for export_dir, dataset_name in zip(export_dirs, datasets_names):
        dataset.add_dir(
            dataset_dir=export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            tags=dataset_name,
        )
    return dataset


def add_prediction(dataset, predictions_path, label_ext=".png"):
    with dataset.save_context() as context:
        for sample in dataset.match_tags("test").select_fields("filepath"):
            p = Path(sample.filepath)
            # go up two level to get the dataset root dir
            # dataset_root = p.parents[2]
            lot = p.parts[-3]
            img_name = p.parts[-1]
            mask_rel_path = os.path.join(lot, img_name.replace(".jpg", label_ext))
            # rel_path = str(p.relative_to(dataset_root)).replace(".jpg", label_ext)
            mask_path = os.path.abspath(os.path.join(predictions_path, mask_rel_path))
            if os.path.exists(mask_path):
                sample["prediction"] = fo.Segmentation(mask_path=mask_path)
                context.save(sample)
    dataset.add_dynamic_sample_fields()
    dataset.save()


def evaluate(dataset: fo.Dataset):
    # dataset.default_mask_targets = {
    #     0: "background",
    #     255: "mp"
    # }

    view = dataset.match_tags("test")
    results = view.evaluate_segmentations(
        "prediction",
        gt_field="ground_truth",
        eval_key="eval_simple",
        mask_targets={
            0: "background",
            255: "mp"
        }
    )

    # Get a sense for the per-sample variation in likeness
    print("Accuracy range: (%f, %f)" % dataset.bounds("eval_simple_accuracy"))
    print("Precision range: (%f, %f)" % dataset.bounds("eval_simple_precision"))
    print("Recall range: (%f, %f)" % dataset.bounds("eval_simple_recall"))

    # Print a classification report
    results.print_report()
    view.default_mask_targets = {255: 'mp'}
    return view


def main(dataset_root_dir, predictions_dir, dataset_name="fo_eval_dataset", remote=True, eval_bool=False):
    if dataset_name in fo.list_datasets():
        print("Dataset already exists: load it")
        dataset = fo.load_dataset(dataset_name)
    else:
        print("Dataset do not exist: create it")
        dataset = create_persistent_dataset(dataset_root_dir, dataset_name)

    if eval_bool:
        add_prediction(dataset, predictions_dir)
        view = evaluate(dataset)
        session = fo.launch_app(view=view, remote=remote)
    else:
        session = fo.launch_app(dataset, remote=remote)
    # Blocks execution until the App is closed
    session.wait(-1)


if __name__ == '__main__':
    # dataset_root_dir = "data/processed/generate_annotated_dataset"
    # predictions_path = "data/processed/mmsegmentation/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/pred"
    # main(dataset_root_dir, predictions_path, remote=False, eval_bool=True)
    fire.Fire(main)
