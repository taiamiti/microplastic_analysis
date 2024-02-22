import tempfile

import fire
import open_clip

from PIL import ImageFile
from tqdm import tqdm

from src.data_prep.embeddings import compute_image_embeddings
import fiftyone as fo
import os
from pathlib import Path
from fiftyone import ViewField as F
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def illu_corr_gray(L, kernel_size=7, ds_factor=3):
    L_downscaled = cv2.resize(L, (L.shape[1] // 3, L.shape[0] // 3))
    illu_downscaled = cv2.medianBlur(L_downscaled, kernel_size)
    # illu_downscaled = cv2.GaussianBlur(L_downscaled, kernel_size)
    illu = cv2.resize(illu_downscaled, (L.shape[1], L.shape[0]))
    newL = L / 255 - illu / 255 + np.mean(illu) / 255  # operations must be done at double precision level and not
    newL[newL > 1] = 1
    newL[newL < 0] = 0
    return (newL * 255).astype(np.uint8)


def img_correction(rgb, kernel_size, ds_factor):
    res = rgb.copy()
    for i in range(3):
        res[:, :, i] = illu_corr_gray(rgb[:, :, i], kernel_size=kernel_size, ds_factor=ds_factor)
    return res


def linear_transfer(src, dest):
    mean_src = src.mean()
    std_src = src.std()

    mean_dest = dest.mean()
    std_dest = dest.std()

    newim = (std_dest / std_src) * (src - mean_src) + mean_dest  # can overflow
    return newim


def create_compo(tri_rgb, blue_rgb, channel=2):
    dest = blue_rgb[:, :, channel] / 255
    src = tri_rgb[:, :, channel] / 255
    newim = linear_transfer(src, dest)
    newim_rgb = blue_rgb.copy()
    newim_rgb[:, :, channel] = (((newim + dest) / 2) * 255).astype(
        np.uint8)  # overflowed value restart at 0 -> yellow for overflow
    return newim_rgb


def create_compo_v2(tri_rgb, blue_rgb, change_scale=1.0, channel=2):
    dest = cv2.cvtColor(blue_rgb, cv2.COLOR_RGB2GRAY) / 255
    src = tri_rgb[:, :, channel] / 255
    newim = linear_transfer(src, dest)
    newim[newim > 1] = 1
    newim[newim < 0] = 0
    diff = newim - dest
    diff_pos = diff.copy()
    diff_pos[diff_pos < 0] = 0
    diff_neg = diff.copy()
    diff_neg[diff_neg > 0] = 0
    moy = (newim + dest) / 2
    newim_rgb = blue_rgb.copy()
    newim_rgb[:, :, 0] = np.clip((moy + change_scale * diff_pos) * 255, 0, 255)
    newim_rgb[:, :, 1] = np.clip(moy * 255, 0, 255)
    newim_rgb[:, :, 2] = np.clip((moy - change_scale * diff_neg) * 255, 0, 255)
    if channel == 0:
        newim_rgb = newim_rgb[:, :, ::-1]
    return newim_rgb


def create_img_composite(tri_fpath, dapi_fpath):
    tri_bgr = cv2.imread(tri_fpath)
    dapi_bgr = cv2.imread(dapi_fpath)
    tri_bgr_corr = img_correction(tri_bgr, kernel_size=51, ds_factor=6)
    dapi_bgr_corr = img_correction(dapi_bgr, kernel_size=51, ds_factor=6)
    compo = create_compo_v2(tri_bgr_corr, dapi_bgr_corr, channel=0, change_scale=1.1)
    return compo


def create_and_export_comp_dataset(input_dataset: str, export_dir: str):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.FiftyOneDataset,
        dataset_dir=input_dataset,
        labels_path=None,
    )

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    with tempfile.TemporaryDirectory() as tmpdirname:
        comp_dataset = create_composite_dataset(dataset, model, preprocess, tmpdirname)
        comp_dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            rel_dir=Path(export_dir).absolute(),
            # move data
            export_media="move"
        )


def create_composite_dataset(dataset: fo.Dataset, model, preprocess, tmpdirname) -> fo.Dataset:
    new_samples = []
    obs_ids = set(dataset.values('obs_id'))
    for obs_id in tqdm(obs_ids):
        tri_sample = dataset.match((F('obs_id') == obs_id) & (F('filter') == 'TRI')).first()
        dapi_sample = dataset.match((F('obs_id') == obs_id) & (F('filter') == 'DAPI')).first()

        # create composite and save
        compo = create_img_composite(tri_sample.filepath, dapi_sample.filepath)
        filepath = os.path.join(tmpdirname, tri_sample.filepath.replace('TRI', 'COMP'))
        os.makedirs(tmpdirname, exist_ok=True)
        cv2.imwrite(filepath, compo)

        # add embeddings
        embeddings = compute_image_embeddings(model, preprocess, filepath).cpu().detach().numpy().tolist()[0]

        # update sample
        sample = tri_sample.copy()
        sample['filepath'] = filepath
        sample['filter'] = "COMP"
        sample['embeddings'] = embeddings
        new_samples.append(sample)
    comp_dataset = fo.Dataset()
    comp_dataset.add_samples(new_samples)
    return comp_dataset


if __name__ == "__main__":
    fire.Fire(create_and_export_comp_dataset)

