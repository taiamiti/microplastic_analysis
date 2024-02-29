#!/usr/bin/env python
# coding: utf-8
import fiftyone as fo
import os

from enum import IntEnum
from fiftyone import ViewField as F


def load_dataset_from_multiple_sources(datasets_paths):
    dataset = fo.Dataset()
    for dataset_path in datasets_paths:
        dataset_name = os.path.basename(dataset_path)
        dataset.add_dir(
            dataset_dir=dataset_path,
            dataset_type=fo.types.FiftyOneDataset,
            tags=dataset_name,
        )
    return dataset


class EvalProtocol(IntEnum):
    BENI_INTRA_INTER_ILE = 1
    BENI_INTRA_ILE = 2
    SED_INTRA_INTER_ILE = 3
    SED_BENI_INTRA_INTER_ILE = 4
    TRAIN_TEST = 5
    UNLABELLED = 6


def get_data_dict(fo_dataset, subset, protocol):
    assert subset in ["train", "test", "unlabelled"], "Wrong subset value"
    if protocol == EvalProtocol.BENI_INTRA_INTER_ILE:
        if subset == "train":
            dataset_view = (fo_dataset.match_tags("train")
                            .match(F('island') == 'TAK')
                            .match(F('sample_type').is_in(("BENI", "CBENI"))))
        else:
            dataset_view = fo_dataset.match_tags("test").match(F('sample_type').is_in(("BENI", "CBENI")))
    elif protocol == EvalProtocol.BENI_INTRA_ILE:
        dataset_view = fo_dataset.match_tags(subset).match(F('sample_type').is_in(("BENI", "CBENI")))
    elif protocol == EvalProtocol.SED_INTRA_INTER_ILE:
        if subset == "train":
            dataset_view = fo_dataset.match_tags("train").match_tags("lot1-20-04-2023-sediments")
        else:
            dataset_view = fo_dataset.match_tags("test").match(F('sample_type').is_in(("SED", "CSED")))
    elif protocol == EvalProtocol.SED_BENI_INTRA_INTER_ILE:
        if subset == "train":
            dataset_view = fo_dataset.match_tags("train").match_tags(["lot1-20-04-2023-benitiers",
                                                                      "lot1-20-04-2023-sediments"])
        else:
            dataset_view = fo_dataset.match_tags("test").match(F('sample_type').is_in(("BENI", "CBENI", "SED", "CSED")))
    elif protocol == EvalProtocol.TRAIN_TEST:
        dataset_view = fo_dataset.match_tags(subset)
    elif protocol == EvalProtocol.UNLABELLED:
        dataset_view = fo_dataset.match_tags("unlabelled")
        return [{"img": sample.filepath} for sample in dataset_view]
    else:
        raise ValueError("Wrong protocol, must be of type EvalProtocol")
    sel_files = [{"img": sample.filepath, "seg": sample.ground_truth.mask_path}
                 for sample in dataset_view if os.path.exists(sample.ground_truth.mask_path)]
    return sel_files


def get_relative_img_paths(ds, subset, protocol, fullpath_prefix):
    return [item["img"].replace(fullpath_prefix + "/", "") for item in get_data_dict(ds, subset, protocol)]


def write_annot_file(file_list, save_dir, subset, protocol):
    save_name = "{}_{}".format(subset, str(protocol).replace(".", "_") + ".txt")
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(save_path, 'w') as fp:
        fp.write('\n'.join(file_list))


def main(ds_root, save_dir):
    ds_paths = [os.path.join(ds_root, ds_path) for ds_path in os.listdir(ds_root)]
    ds = load_dataset_from_multiple_sources(ds_paths)
    # ds.match_tags("unlabelled")
    # ds.match_tags("test").match(F('sample_type').is_in(("BENI", "CBENI")))
    # ds.count_sample_tags()
    #
    # beni_view = ds.match(F('sample_type').is_in(("BENI", "CBENI"))).filter_labels("detections", F("score") > 0.2)
    # sed_view = ds.match(F('sample_type').is_in(("SED", "CSED"))).filter_labels("detections", F("score") > 0.35)
    # merged_view = beni_view + sed_view
    # merged_view_clean = merged_view.match_tags('bad_gt', bool=False)

    # session = fo.launch_app(ds, auto=False)
    # session.open_tab()
    #
    # view_filtered_out = (ds.match(F('sample_type').is_in(("BENI", "CBENI")))
    #                      .filter_labels("detections", F("score") > 0.2, only_matches=False)
    #                      .match(F('detections.detections').length() == 0)
    #                      )
    #
    # session.view = view_filtered_out
    # merged_view_clean.match_tags("test").match(F('sample_type').is_in(("BENI", "CBENI")))

    ds.export(
        export_dir=save_dir,
        dataset_type=fo.types.ImageSegmentationDirectory,
        label_field='detections',
        export_media=True,
        rel_dir=os.path.abspath(ds_root)
    )

    fullpath_prefix = os.path.abspath(ds_root)
    for protocol in EvalProtocol:
        if protocol == EvalProtocol.UNLABELLED:
            file_list = get_relative_img_paths(ds, "unlabelled", protocol, fullpath_prefix)
            write_annot_file(file_list, save_dir, "unlabelled", protocol)
        for subset in ["train", "test"]:
            #         file_list = get_relative_img_paths(merged_view_clean, subset, protocol, fullpath_prefix)
            file_list = get_relative_img_paths(ds, subset, protocol, fullpath_prefix)
            write_annot_file(file_list, save_dir, subset, protocol)


if __name__ == '__main__':
    ds_root_ = "data/processed/generate_annotated_dataset/"
    save_dir_ = "data/processed/prepare_dataset_for_openmmseg/"
    main(ds_root_, save_dir_)

