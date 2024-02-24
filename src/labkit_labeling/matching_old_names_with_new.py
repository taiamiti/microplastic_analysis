import os
import shutil

import pandas as pd

import torch
from PIL import Image
import open_clip
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

"""
Copy masks to `data/processed/labkitinference` folder using either name matching or image content matching 
of source images
"""


def compute_image_embeddings(model, preprocess, fpath):
    image = preprocess(Image.open(fpath)).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def compute_embeddings(filepaths, model, preprocess):
    features_list = []
    for filepath in tqdm(filepaths):
        try:
            features = compute_image_embeddings(model, preprocess, filepath)
        except:
            print(filepath)
            features = np.zeros((1, 512))
        features_list.append(features)
    # Compute embeddings
    embeddings = torch.cat(features_list).cpu().detach().numpy()
    return embeddings


def matching_between_two_image_sets(fpath1, fpath2, min_thresh=0.0001):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    X1 = compute_embeddings(fpath1, model, preprocess)
    X2 = compute_embeddings(fpath2, model, preprocess)
    dist = euclidean_distances(X1, X2)
    new_order = np.argmin(dist, axis=1)
    mins = dist.min(axis=1)
    pairs = [(x, y) for x, y in zip(fpath1, [fpath2[i] for i in new_order])]
    # filter only true match pairs
    valid_index = mins < min_thresh
    return pairs, valid_index


def align_names_between_two_image_sets(old_dataset, old_labelled_dataset, renamed_dataset, renamed_labelled_dataset):
    fpath1 = [str(s) for s in renamed_dataset.rglob("*.jpg")]
    fpath2 = [str(s) for s in old_dataset.rglob("*.jpg")]
    pairs, valid_index = matching_between_two_image_sets(fpath1, fpath2)
    os.makedirs(renamed_labelled_dataset, exist_ok=True)
    for (renamed, old), is_valid in zip(pairs, valid_index):
        if not is_valid:
            print("invalid pair")
            print((renamed, old))
            continue
        try:
            # labelled data are png instead of jpg
            new_filepath = str(renamed_labelled_dataset / Path(renamed).name).replace('.jpg', '.png')
            old_labelled_path = old.replace(str(old_dataset), str(old_labelled_dataset)).replace('.jpg', '.png')
            shutil.copy(old_labelled_path, new_filepath)
        except shutil.Error as err:
            logger.error(err)
            logger.error(renamed, old)


# def main_lot1_beni():
#     # for each file in dset1, try to find its matching pair in dset2
#     renamed_dataset = Path("data/processed/data_to_annotate/lot1-20-04-2023-benitiers")
#     old_dataset = Path("data/processed/data_to_annotate/lot2")
#     old_labelled_dataset = Path("data/processed/annotated_data/lot1")
#     renamed_labelled_dataset = Path("data/processed/labkitinference/lot1-20-04-2023-benitiers")
#
#     align_names_between_two_image_sets(old_dataset, old_labelled_dataset, renamed_dataset, renamed_labelled_dataset)
#
#
# def main_lot1_sed():
#     # for each file in dset1, try to find its matching pair in dset2
#     renamed_dataset = Path("data/processed/create_tasks/lot1-20-04-2023-sediments")
#     old_dataset = Path("data/processed/data_to_annotate/lot2")
#     old_labelled_dataset = Path("data/processed/annotated_data/lot1")
#     renamed_labelled_dataset = Path("data/processed/labkitinference/lot1-20-04-2023-sediments")
#
#     align_names_between_two_image_sets(old_dataset, old_labelled_dataset, renamed_dataset, renamed_labelled_dataset)
#
#
# def main_lot2_nacl():
#     # for each file in dset1, try to find its matching pair in dset2
#     renamed_dataset = Path("data/processed/create_composite/lot2-30-05-2023-tak_nacl")
#     old_dataset = Path("data/processed/data_to_annotate/lot2")
#     old_labelled_dataset = Path("data/processed/annotated_data/lot2")
#     renamed_labelled_dataset = Path("data/processed/labkitinference/lot2-30-05-2023-tak_nacl2")
#
#     align_names_between_two_image_sets(old_dataset, old_labelled_dataset, renamed_dataset, renamed_labelled_dataset)
#
#
# def main_lot2_nai():
#     # for each file in dset1, try to find its matching pair in dset2
#     renamed_dataset = Path("data/processed/create_composite/lot2-30-05-2023-tak_nai")
#     old_dataset = Path("data/processed/data_to_annotate/lot2")
#     old_labelled_dataset = Path("data/processed/annotated_data/lot2")
#     renamed_labelled_dataset = Path("data/processed/labkitinference/lot2-30-05-2023-tak_nai")
#
#     align_names_between_two_image_sets(old_dataset, old_labelled_dataset, renamed_dataset, renamed_labelled_dataset)


def copy_matching_name(dataset_dir, annotated_dir, new_annotated_dir):
    """
    When annotating, the data structure is modified to match task partitioning. It is easier to convert the labelled
    masks back to the original data structure for further manipulation (ie: loading with fiftyone).
    This function works for unique names only

    :param dataset_dir:
    :param annotated_dir:
    :param new_annotated_dir:
    :return:
    """
    fpath1 = [s for s in Path(dataset_dir).rglob("*.jpg")]
    fpath2 = [s for s in Path(annotated_dir).rglob("*.png")]
    os.makedirs(new_annotated_dir, exist_ok=True)
    for f1 in fpath1:
        for f2 in fpath2:
            if f1.stem == f2.stem:
                try:
                    # labelled data are png instead of jpg
                    new_filepath = str(f1).replace(dataset_dir, new_annotated_dir).replace(".jpg", ".png")
                    old_labelled_path = str(f2)
                    shutil.copy(old_labelled_path, new_filepath)
                    logger.info("copy {} to {} successful".format(old_labelled_path, new_filepath))
                except shutil.Error as err:
                    logger.error(err)
                    logger.error(old_labelled_path, new_filepath)


def main_lot3():
    # for each file in dset1, try to find its matching pair in dset2
    dataset_dir = "data/processed/create_composite/lot3-08-06-2023-benitiers/data"
    annotated_dir = "data/processed/annotated_data/lot3-08-06-2023-benitiers"
    new_annotated_dir = "data/processed/labkitinference/lot3-08-06-2023-benitiers"
    copy_matching_name(dataset_dir, annotated_dir, new_annotated_dir)


def main_lot4():
    for dataset_name in ["lot4-28-06-2023-sediments-part1",
                         "lot4-28-06-2023-sediments-part2",
                         "lot4-28-06-2023-sediments-part3"]:
        # for each file in dset1, try to find its matching pair in dset2
        dataset_dir = f"data/processed/create_composite/{dataset_name}/data"
        annotated_dir = "/data/processed/annotated_data/lot4"
        new_annotated_dir = f"data/processed/labkitinference/{dataset_name}"
        copy_matching_name(dataset_dir, annotated_dir, new_annotated_dir)


def main_reannot_lot1_4_beni():
    for dataset_name in ["lot1-20-04-2023-benitiers",
                         "lot3-08-06-2023-benitiers"]:
        # for each file in dset1, try to find its matching pair in dset2
        dataset_dir = f"data/processed/create_composite/{dataset_name}/data"
        annotated_dir = "data/processed/annotated_data/lot1_lot4_review_beni"
        new_annotated_dir = f"data/processed/labkitinference/{dataset_name}"
        copy_matching_name(dataset_dir, annotated_dir, new_annotated_dir)


def main_reannot_lot1_4_sed():
    for dataset_name in ["lot1-20-04-2023-sediments",
                         "lot2-30-05-2023-tak_nacl",
                         "lot2-30-05-2023-tak_nai",
                         "lot2-30-05-2023-tak_nai-part2",
                         "lot4-28-06-2023-sediments-part1",
                         "lot4-28-06-2023-sediments-part2",
                         "lot4-28-06-2023-sediments-part3"]:
        # for each file in dset1, try to find its matching pair in dset2
        dataset_dir = f"data/processed/create_composite/{dataset_name}/data"
        annotated_dir = "data/processed/annotated_data/lot1_lot4_review_sed"
        new_annotated_dir = f"data/processed/labkitinference/{dataset_name}"
        copy_matching_name(dataset_dir, annotated_dir, new_annotated_dir)


def main_lot5_10():
    # will replace already annotated data for correction
    for dataset_name in ["lot5-04-07-2023-benitiers-part1",
                         "lot5-04-07-2023-benitiers-part2",
                         "lot6-12-08-2023-eau-horizontal",
                         "lot6-12-08-2023-eau-vertical",
                         "lot7-28-09-2023-benitiers",
                         "lot8-28-09-2023-benitiers",
                         "lot9-09-10-2023-benitiers",
                         "lot10-09-10-2023-benitiers"]:
        # for each file in dset1, try to find its matching pair in dset2
        dataset_dir = f"data/processed/create_composite/{dataset_name}/data"
        annotated_dir = "data/processed/annotated_data/lot5_10"
        new_annotated_dir = f"data/processed/labkitinference/{dataset_name}"
        copy_matching_name(dataset_dir, annotated_dir, new_annotated_dir)


if __name__ == "__main__":
    # main_lot3()
    # main_lot4()
    # main_lot5_10()
    # main_reannot_lot1_4_beni()
    main_reannot_lot1_4_sed()

