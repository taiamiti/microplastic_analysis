import dataclasses
import os
import shutil
import sys

from pathlib import Path

import fire
import numpy as np
import re

from PIL import Image
from tqdm import tqdm

import time

from src.data_prep.custom_logger import loguru_setup
import easyocr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import fiftyone as fo
from fiftyone import ViewField as F

from src.data_prep.custom_exceptions import FileNotNamedCorrectly, MultipleNamingConvention
from src.data_prep.data_utils import MetaData, get_obs_id, DataPrior
from src.data_prep.embeddings import compute_embeddings, load_embedding_centers
from collections import Counter, OrderedDict

# LOG_PATH = "./logs/{}.log".format(Path(__file__).stem)
# os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
# logger.basicConfig(filename=LOG_PATH,
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logger.DEBUG)

LOG_PATH = "./logs/{}.log".format(Path(__file__).stem)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = loguru_setup(LOG_PATH)


def parse_zoom_text_from_image(reader, image_path):
    img = Image.open(image_path)
    width, height = img.size
    left = int(0.8 * width)
    top = int(0.9 * height)
    right = width
    bottom = height
    cropped = img.crop((left, top, right, bottom))
    newsize = (2 * cropped.size[0], 2 * cropped.size[1])
    cropped = cropped.resize(newsize)  # make bigger after crop for better text det
    cr = np.array(cropped)
    result = reader.readtext(cr[:, :, ::-1])  # make it bgr as opencv image is expected
    if len(result) > 0:
        text = result[0][1]
    else:
        text = 'no text'
    return text


def infer_image_id_from_order(num_imgs):
    assert num_imgs % 4 == 0, "DataPrior is invalid, number of images must be divisible by 4"
    x = np.ones((num_imgs // 4, 4), dtype=int)
    int_ids = x.cumsum(axis=0).flatten().tolist()
    return ["{:04d}".format(int_id) for int_id in int_ids]


def create_dataset(image_paths: list):
    reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
    samples = []
    for image_path in tqdm(image_paths):
        img_metadata = fo.ImageMetadata.build_for(image_path)
        extra_metadata = MetaData(image_path)
        extra_metadata = dataclasses.asdict(extra_metadata)
        extra_metadata["zoom"] = parse_zoom_text_from_image(reader, image_path)
        sample = fo.Sample(filepath=image_path, metadata=img_metadata)
        for k, v in extra_metadata.items():
            sample[k] = v
        samples.append(sample)

    dataset = fo.Dataset()
    dataset.add_samples(samples)
    dataset.add_dynamic_sample_fields()
    return dataset


def transform_dataset(dataset: fo.Dataset, prior: DataPrior, embedding_center_path: str):
    embedding_centers, center_label = load_embedding_centers(embedding_center_path)
    # override image_id with new id
    if prior == DataPrior.CONSECUTIVE:
        image_ids = infer_image_id_from_order(len(dataset))
        dataset.set_values('image_id', image_ids)

    # get observation ids after valid image_id
    dataset.set_values('obs_id', [get_obs_id(sample) for sample in dataset])

    # filter valid observations
    logger.info("Filter invalid samples")
    dataset, invalid_filepaths = filter_out_invalid_samples(dataset)
    # save invalid_obs_id as an artefact

    logger.info("Add embeddings")
    dataset = add_embeddings(dataset)

    # override filter with correct filter
    logger.info("Add or Fix filter")
    dataset = add_or_fix_filter(dataset, embedding_centers, center_label)

    dataset.save()  # save change to the database
    return dataset, invalid_filepaths


def filter_out_bad_zoom(dataset):
    match_cdt = (F("zoom").is_in(("500 pm", "200 pm"))
                 & (F("metadata.width") == 1920)
                 & (F("metadata.height") == 1200))
    bad_filepaths = [sample.filepath for sample in dataset.match(~match_cdt)]
    for bad_filepath in bad_filepaths:
        logger.warning(bad_filepath + ": discarded because bad zoom or image size")
    return dataset.match(match_cdt), bad_filepaths


def filter_out_extra(dataset):
    bad_filepaths = [sample.filepath for sample in dataset.match(F("extra") != "")]
    for bad_filepath in bad_filepaths:
        logger.warning(bad_filepath + ": discarded because CUT or BIS image")
    return dataset.match(F("extra") == ""), bad_filepaths


def filter_4_images_per_obs(dataset):
    obs_ids = [sample.obs_id for sample in dataset]
    counter = Counter(obs_ids)
    invalid_obs_id = []
    for unique_obs_id in counter:
        num_obs = counter[unique_obs_id]
        if num_obs != 4:
            logger.warning(unique_obs_id + ': discarded because found {} instead of 4'.format(num_obs))
            invalid_obs_id.append(unique_obs_id)
    invalid_filepaths = [sample.filepath for sample in dataset if sample.obs_id in invalid_obs_id]
    valid_dataset = dataset.match(~F("filepath").is_in(invalid_filepaths))
    return valid_dataset, invalid_filepaths


def filter_out_invalid_samples(dataset):
    dataset, badfiles1 = filter_out_bad_zoom(dataset)
    dataset, badfiles2 = filter_out_extra(dataset)
    valid_dataset, badfiles3 = filter_4_images_per_obs(dataset)
    return valid_dataset, badfiles1 + badfiles2 + badfiles3


def add_embeddings(dataset):
    image_paths = [sample.filepath for sample in dataset]
    embeddings = compute_embeddings(image_paths)
    dataset.set_values("embeddings", embeddings.tolist())
    return dataset


def infer_filter_from_imgs(embeddings, embedding_centers, center_label):
    assert embeddings.shape[0] == 4, "embeddings size = {} but should have 4 observations".format(embeddings.shape[0])
    # suppose that embeddings have 4 observations
    cost_matrix = cdist(embeddings, embedding_centers, 'euclidean')  # color_data_scaled
    row, col = linear_sum_assignment(cost_matrix)
    assign_filter = col.astype(int).tolist()
    new_filter = np.array(center_label)[assign_filter]
    return new_filter


def add_or_fix_filter2(dataset: fo.Dataset, embedding_centers, center_label):
    # Global clustering does not work as shown in scatterplot.
    # However, using best matching per sample (4 filters / images) to the 4 clusters
    # we obtain a good filter identification.
    obs_ids = set([sample.obs_id for sample in dataset])
    with dataset.save_context() as context:
        for obs_id in tqdm(obs_ids):
            sel_sample_ids = [sample.id for sample in dataset if sample.obs_id == obs_id]
            embeddings = np.array([sel_sample.embeddings for sel_sample in dataset[sel_sample_ids]])
            filters = infer_filter_from_imgs(embeddings, embedding_centers, center_label)
            for sample, filter_ in zip(dataset[sel_sample_ids], filters):
                sample["filter"] = filter_
                context.save(sample)
    return dataset


def add_or_fix_filter(dataset: fo.Dataset, embedding_centers, center_label):
    # Global clustering does not work as shown in scatterplot.
    # However, using best matching per sample (4 filters / images) to the 4 clusters
    # we obtain a good filter identification.
    results = OrderedDict.fromkeys(dataset.values('id'))
    obs_ids_unique = set(dataset.values('obs_id'))
    for obs_id in tqdm(obs_ids_unique):
        sel_view = dataset.match(F('obs_id') == obs_id)
        embeddings = np.array([sel_sample.embeddings for sel_sample in sel_view])
        filters = infer_filter_from_imgs(embeddings, embedding_centers, center_label)
        for id_, filter_ in zip(sel_view.values('id'), filters):
            results[id_] = filter_
    dataset.set_values("filter", list(results.values()))
    return dataset


# ## Export dataset with proper naming

def export_with_proper_naming(dataset: fo.Dataset, export_dir: str):
    import tempfile

    # first create a new images in a temp dir
    with tempfile.TemporaryDirectory() as tmpdirname:
        for sample in tqdm(dataset):
            new_filepath = os.path.join(tmpdirname, get_new_name(sample))
            os.makedirs(tmpdirname, exist_ok=True)
            try:
                shutil.copy(sample.filepath, new_filepath)
            except shutil.Error as err:
                logger.error(err)
                logger.error(sample.filepath)
                sys.exit(1)
            sample["filepath"] = new_filepath
            sample.save()

        # export dataset by moving files to desired destination
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.FiftyOneDataset,  # Any subclass of `fiftyone.types.Dataset` is supported
            label_field=None,
            # rel_dir=tmpdirname.absolute(),  # it is important otherwise the image subdir structure is lost and
            # flattened
            export_media="move"
        )
        time.sleep(3)


def get_new_name(sample):
    new_name = f"{sample.obs_id}_{sample.filter}_{sample.zoom}.jpg"
    return new_name


def infer_data_naming_prior(fname):
    if re.findall(r"\(\d+\)", fname):  # match (i)
        return DataPrior.PARTIAL_RENAMED
    elif ("dapi" in fname.lower()) or ("cy2" in fname.lower()) or "nat" in fname.lower() or "tri" in fname.lower():
        return DataPrior.RENAMED
    elif re.findall(r"_\d{4}.jpg", fname):
        return DataPrior.CONSECUTIVE
    else:
        raise FileNotNamedCorrectly(f"Wrong naming prior {fname}, discard this file")


def infer_data_naming_prior_for_list(fnames):
    priors = []
    valid_fnames = []
    bad_fnames = []
    for fname in fnames:
        try:
            prior = infer_data_naming_prior(fname)
            priors.append(prior)
            valid_fnames.append(fname)
        except FileNotNamedCorrectly:
            bad_fnames.append(fname)
            continue
    if len(set(priors)) > 1:
        for prior in set(priors):
            logger.error([fname for fname, p in zip(fnames, priors) if p == prior])
        raise MultipleNamingConvention(f"Multiple naming convention")
    elif len(priors) == 0:
        raise ValueError("input list is empty")
    else:
        return priors[0], valid_fnames, bad_fnames


def main_debug():
    embedding_center_path = "data/processed/compute_embedding_filter_centers/embedding_centers_lot2.json"
    for lot in os.listdir("data/unit_test/raw"):
        image_dir = f"data/unit_test/raw/{lot}"
        out_root_dir = f"data/unit_test/processed/ingest_data/{lot}"
        if os.path.exists(out_root_dir):
            logger.info("Output already exist, overwrite")
        main(image_dir, embedding_center_path, out_root_dir)


def main(dataset_local_path, embedding_center_path, ingested_data_path):
    image_paths = sorted([str(fp) for fp in Path(dataset_local_path).rglob("*.jpg")])
    try:
        prior, valid_paths, bad_paths = infer_data_naming_prior_for_list(image_paths)
    except MultipleNamingConvention as error:
        logger.exception(error)
        logger.error(f"skipping ingest data for {dataset_local_path}")
        raise MultipleNamingConvention
    logger.info("Create dataset")
    ds = create_dataset(valid_paths)
    logger.info("Transform dataset")
    ds, invalid_paths = transform_dataset(ds, prior, embedding_center_path)
    invalid_paths = invalid_paths + bad_paths
    logger.info(f"Invalid paths: {invalid_paths}")
    logger.info("Export dataset")
    export_with_proper_naming(ds, ingested_data_path)


if __name__ == "__main__":
    # try:
    #     main_debug()
    # except AssertionError:
    #     logger.error("Abort program, fix the issue then retry", exc_info=True)
    #     sys.exit(1)
    fire.Fire(main)
