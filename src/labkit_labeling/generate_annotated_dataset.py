#!/usr/bin/env python
# coding: utf-8
"""
Generate annotated dataset from binary segmentation masks :
- create fiftyone dataset of type segmentation masks
- convert masks to fiftyone detections (bbox + instance mask)
- optional : convert detections to instance segmentation polylines (for cvat)
- compute detection attributes :
    - score based on contrast
    - connected component properties (feret diameter, particule type, ...)
    - RGB values
- tag dataset with split tags (train, test, unlabelled)
"""

import os

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import distance

import fiftyone as fo

from tqdm import tqdm


@dataclass
class Detection:
    contour: np.array
    mp_shape: str = field(init=False)
    rect: tuple = field(init=False)  # x,y,w,h
    area: float = field(init=False)
    feret_diameter: float = field(init=False)
    feret_degree: float = field(init=False)
    feret_pointa: tuple = field(init=False)
    feret_pointb: tuple = field(init=False)
    circularity: float = field(init=False)
    roundness: float = field(init=False)
    perimeter: float = field(init=False)

    def __post_init__(self):
        self.rect = tuple(cv2.boundingRect(self.contour))
        self.area = cv2.contourArea(self.contour)
        self.feret_diameter, self.feret_degree, self.feret_pointa, self.feret_pointb = self._compute_feret(self.contour)
        self.roundness = 4 * self.area / (np.pi * np.power(self.feret_diameter, 2))
        self.perimeter = cv2.arcLength(self.contour, False)
        self.circularity = 4 * np.pi * (self.area / np.power(self.perimeter, 2))
        self.mp_shape = self._get_mp_shape(self.circularity)

    @staticmethod
    def _get_mp_shape(circularity):
        mp_shape = "None"

        if min(0, 0.3) <= circularity < max(0, 0.3):
            mp_shape = "Fibers"
        elif min(0.3, 0.7) <= circularity < max(0.3, 0.7):
            mp_shape = "Fragments"
        else:
            mp_shape = "Particles"
        return mp_shape

    @staticmethod
    def _compute_feret(contour):
        # feret diameters : distance between all points in a contour
        reshaped_contour = contour.reshape((contour.shape[0], contour.shape[-1]))
        feret_diameters = distance.cdist(reshaped_contour, reshaped_contour, 'sqeuclidean')

        ## max val
        feret_diameter = feret_diameters.max()

        # feret degree :
        (a, b) = np.unravel_index(feret_diameters.argmax(), feret_diameters.shape)
        point_a = reshaped_contour[a]
        point_b = reshaped_contour[b]

        deltaY = point_a[1] - point_b[1]
        deltaX = point_a[0] - point_b[0]
        angleInDegrees = np.arctan2(deltaY, deltaX) * 180 / np.pi
        return feret_diameter, angleInDegrees, tuple(point_a), tuple(point_b)


def mp_act(mask_image):
    # use only external contours for instance segmentation otherwise issues with holes
    contour_list_cv2, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for index, contour in enumerate(contour_list_cv2):
        det = Detection(contour)
        dets.append(det)
    return dets


def filter_dets(detections, min_area=40, max_area=400 * 400):
    ret = []
    for i, det in enumerate(detections):
        if (det.area < min_area) or (det.area > max_area):
            continue
        ret.append(det)
    return ret


def segment_sample(mask_path) -> (fo.Detections, fo.Polylines):
    bw = 255 * cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # add black border to avoid issues with CC at the border
    bw[0, :] = 0
    bw[-1, :] = 0
    bw[:, 0] = 0
    bw[:, -1] = 0
    img_height, img_width = bw.shape[:2]
    dets_ = mp_act(bw)
    dets = filter_dets(dets_)

    return dets_to_fodetections(dets, img_height, img_width), dets_to_fopolylines(dets, img_height, img_width)


def dets_to_fodetections(dets, img_height, img_width):
    # Convert detections to FiftyOne format
    detections = []
    for det in dets:
        label = det.mp_shape

        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        bounding_box = [det.rect[0] / img_width,
                        det.rect[1] / img_height,
                        det.rect[2] / img_width,
                        det.rect[3] / img_height]
        img = np.zeros((det.rect[3], det.rect[2], 3))
        if det.contour.shape[0] < 3:
            continue
        mask = cv2.drawContours(img, [det.contour - np.array([[[det.rect[0], det.rect[1]]]])], -1,
                                color=(255, 255, 255), thickness=cv2.FILLED)[:, :, 0]
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)  # avoid degenerate case
        # binary_fill_holes(bw[det.rect[1]:det.rect[1] + det.rect[3], det.rect[0]:det.rect[0] + det.rect[2]]
        #                      .astype(bool))
        detections.append(
            fo.Detection(label=label,
                         bounding_box=bounding_box,
                         mask=mask,
                         area=det.area,  # custom attribute
                         feret_diameter=det.feret_diameter,  # custom attribute
                         feret_degree=det.feret_degree,
                         feret_pointa=det.feret_pointa,
                         feret_pointb=det.feret_pointb,
                         circularity=det.circularity,
                         roundness=det.roundness,
                         perimeter=det.perimeter,
                         )
        )
    return fo.Detections(detections=detections)


def dets_to_fopolylines(dets, img_height, img_width):
    # Convert detections to FiftyOne format
    polylines = []
    for det in dets:
        # A closed, filled polygon with a label
        if det.contour.shape[0] < 3:
            continue
        # if det.mp_shape == "Fibers":
        #     contour = det.contour
        # else:
        #     contour = cv2.convexHull(det.contour)
        contour = det.contour
        contour = resample_contour(contour, det.perimeter)
        # note to regularize the shape between hull and actual shape, resample both hull and shape then make a linear
        # combination of both
        contour = contour / np.array([[[img_width, img_height]]])
        polylines.append(
            fo.Polyline(
                label=det.mp_shape,
                points=contour.reshape(1, -1, 2).tolist(),
                closed=True,
                filled=True,
            )
        )
    return fo.Polylines(polylines=polylines)


def resample_contour(contour, perimeter, pixel_spacing=7):
    M = int(perimeter / pixel_spacing)
    if M < 3:
        return contour
    newt = np.linspace(0, 1, M)
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    t = np.linspace(0, 1, contour.shape[0])  # goes from 0 to 1 with N points
    x_interp = np.interp(newt, t, x)
    y_interp = np.interp(newt, t, y)
    new_contour = np.zeros((x_interp.shape[0], 1, 2))
    new_contour[:, 0, 0] = x_interp
    new_contour[:, 0, 1] = y_interp
    return new_contour


def compute_box_contrast(contrast, bbox, bbox_mask, offset=10):
    img_h, img_w = contrast.shape[:2]
    bbox_x = int(bbox[0] * img_w)
    bbox_y = int(bbox[1] * img_h)
    bbox_h, bbox_w = bbox_mask.shape
    y0, y1 = max(bbox_y - offset, 0), min(bbox_y + bbox_h + offset, img_h)
    x0, x1 = max(bbox_x - offset, 0), min(bbox_x + bbox_w + offset, img_w)
    patch = contrast[y0:y1, x0:x1]
    return (patch.max() - patch.min()) / contrast.max()  # use absolute contrast


def compute_box_rgb(rgb, bbox, bbox_mask):
    img_h, img_w = rgb.shape[:2]
    bbox_x = int(bbox[0] * img_w)
    bbox_y = int(bbox[1] * img_h)
    bbox_h, bbox_w = bbox_mask.shape
    y0, y1 = max(bbox_y, 0), min(bbox_y + bbox_h, img_h)
    x0, x1 = max(bbox_x, 0), min(bbox_x + bbox_w, img_w)
    patch = rgb[y0:y1, x0:x1, :]
    r = patch[:,:,0][bbox_mask].mean()
    g = patch[:, :, 1][bbox_mask].mean()
    b = patch[:, :, 2][bbox_mask].mean()
    return r, g, b


def compute_bg_color(img, mask):
    return img[mask == 0].mean(axis=0)


def compute_fg_color(img, mask):
    return img[mask != 0].mean(axis=0)


def compute_score(bg_color, max_contrast, box_fg_color):
    dif_box = np.sqrt(np.sum((bg_color - box_fg_color) ** 2))
    return int(100 * dif_box / max_contrast)


def score_sample(sample, mask_key="ground_truth", det_key='detections', poly_key='polylines'):
    # add score based on contrast + mean color values
    img = cv2.imread(sample["filepath"])
    mask = cv2.imread(sample[mask_key]["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    bg_color = compute_bg_color(img, mask)
    # fg_color = compute_fg_color(img, mask)
    L = np.sqrt(np.sum((bg_color - img) ** 2, axis=2))
    for det, poly in zip(sample[det_key].detections, sample[poly_key].polylines):
        # add score
        det.score = compute_box_contrast(L, det.bounding_box, det.mask)
        poly.score = det.score
        # add mean color
        b, g, r = compute_box_rgb(img, det.bounding_box, det.mask)
        det.mean_blue, det.mean_green, det.mean_red = r, g, b
        poly.mean_blue, poly.mean_green, poly.mean_red = r, g, b


def add_semantic_labels(dataset: fo.Dataset, labels_path: str, label_ext=".png") -> fo.Dataset:
    # labelled_ds = fo.Dataset.from_dir(
    #     data_path=labels_path,
    #     labels_path=labels_path,
    #     dataset_type=fo.types.ImageSegmentationDirectory,
    # )
    # Use a context to save sample edits in efficient batches
    with dataset.save_context() as context:
        for sample in dataset.select_fields("filepath"):
            basename = os.path.splitext(os.path.basename(sample.filepath))[0] + label_ext
            mask_path = os.path.join(labels_path, basename)
            if os.path.exists(mask_path):
                sample["ground_truth"] = fo.Segmentation(mask_path=mask_path)
                context.save(sample)
            else:
                sample.tags.append("unlabelled")
                context.save(sample)
    # Set default mask targets
    dataset.default_mask_targets = {255: "mp"}
    dataset.save()

    # key_fcn = lambda sample: os.path.basename(sample.filepath)
    # dataset.merge_samples(labelled_ds, key_fcn=key_fcn)

    # return only samples with labels
    return dataset


def add_instance_segmentation(dataset):
    with dataset.save_context() as context:
        for sample in tqdm(dataset):
            filename = os.path.basename(sample["filepath"])
            sample['filename'] = filename
            sample['detections'], sample['polylines'] = segment_sample(sample["ground_truth"]["mask_path"])
            score_sample(sample)
            context.save(sample)


def add_split_as_tag(dataset, test_size=0.3):
    # split into train, val (annotated) test (not annotated)
    # split based on island+station at dataset level
    # origin = ile + station + replica
    from sklearn.model_selection import train_test_split
    from fiftyone import ViewField as F

    origin = ["-".join(l) for l in zip(dataset.values('sample_type'),
                                       dataset.values('island'),
                                       dataset.values('station'),
                                       dataset.values('replica'),
                                       )]
    dataset.set_values('origin', origin)

    ## filter samples that are numerous enough for a given origin
    counts = dataset.count_values("origin")
    keep_names = [name for name, count in counts.items() if count > 2]
    view = dataset.match(F("origin").is_in(keep_names))

    if len(view) < 10:
        test_size = 1.0  # add everything in test

    ## split and tag
    if test_size == 1.0:
        # everything in test set
        view.tag_samples("test")
    elif test_size == 0:
        # everything in training
        view.tag_samples("train")
    else:
        sample_ids = view.values('id')
        origin = view.values('origin')
        sid_train, sid_test, ori_train, ori_test = train_test_split(sample_ids, origin,
                                                                    stratify=origin,
                                                                    test_size=test_size,
                                                                    random_state=42)
        view[sid_train].tag_samples("train")
        view[sid_test].tag_samples("test")
    view.save()


def add_cluster_as_tag():
    pass


def main(dataset_path, labels_path, dataset_export_path):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_path,
        dataset_type=fo.types.FiftyOneDataset,
    )

    # Create the semantic segmentation dataset
    dataset = add_semantic_labels(dataset, labels_path)
    labelled_view = dataset.match_tags("unlabelled", bool=False)
    add_split_as_tag(labelled_view)  # add split tag on labelled data only

    # # Add instance segmentation
    # # Use a context to save sample edits in efficient batches
    add_instance_segmentation(labelled_view)  # transform into instance labelled data only

    # dynamic sample fields do not appear in the app interactive selection tool if the below line is not executed
    dataset.add_dynamic_sample_fields()
    dataset.save()

    # export into fiftyone dataset to have all the dataset fields
    dataset.export(
        export_dir=dataset_export_path,
        dataset_type=fo.types.FiftyOneDataset,
        # rel_dir=os.getcwd(),
        export_media=True
    )

    #
    # # detections with masks allows objects with hole whereas polylines are limited to external contour of shapes
    # # however to convert into cvat we need polylines
    # # from fiftyone.utils.labels import instances_to_polylines
    # # instances_to_polylines(dataset, "detections", "polylines", tolerance=2, filled=False)
    # # the conversion above do not work well, do it manually
    #
    # # Export the instance seg dataset as cvat (have polylines only) to fix annotations on cvat eventually
    # dataset.export(
    #     export_dir=dataset_export_path + '_instance',
    #     dataset_type=fo.types.CVATImageDataset,
    #     label_field="polylines",  # polylines
    #     # rel_dir=os.getcwd(),
    #     export_media=True
    # )


if __name__ == '__main__':
    dataset_names = [
        "lot1-20-04-2023-benitiers",
        "lot1-20-04-2023-sediments",
        "lot2-30-05-2023-tak_nacl",
        "lot2-30-05-2023-tak_nai",
        "lot3-08-06-2023-benitiers",
        "lot4-28-06-2023-sediments-part1",
        "lot4-28-06-2023-sediments-part2",
        "lot4-28-06-2023-sediments-part3",
        "lot5-04-07-2023-benitiers-part1",
        "lot5-04-07-2023-benitiers-part2",
        "lot6-12-08-2023-eau-horizontal",
        "lot6-12-08-2023-eau-vertical",
        "lot7-28-09-2023-benitiers",
        "lot8-28-09-2023-benitiers",
        "lot9-09-10-2023-benitiers",
        "lot10-09-10-2023-benitiers",
    ]
    for dataset_name in dataset_names:
        ds_path_ = f"data/processed/create_composite/{dataset_name}"
        labels_path_ = f"data/processed/labkitinference/{dataset_name}"
        dataset_export_path_ = f"data/processed/generate_annotated_dataset2/{dataset_name}"
        main(ds_path_, labels_path_, dataset_export_path_)
