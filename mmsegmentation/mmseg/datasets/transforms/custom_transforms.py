from typing import Union, Tuple

import numpy as np
from mmcv.transforms import BaseTransform, TRANSFORMS
from mmcv.transforms.utils import cache_randomness
import cv2


@TRANSFORMS.register_module()
class InvertBinaryLabels(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        img = results.get('gt_seg_map', np.zeros(results['img'].shape[:2], dtype=np.uint8))
        img[img == 255] = 1
        img = cv2.GaussianBlur(img, (3, 3), 0)
        results['gt_seg_map'] = img
        return results


@TRANSFORMS.register_module()
class RandomCropForeground(BaseTransform):
    """Random crop the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray, gt_seg_map:np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """
            overlap = 0.25
            yis, xis = np.where(gt_seg_map == 1)
            if len(yis) > 0 and len(xis) > 0:
                sel_pix_y = yis[np.random.randint(0, yis.shape[0])]
                sel_pix_x = xis[np.random.randint(0, xis.shape[0])]
                offset_y = np.random.randint(int(overlap*self.crop_size[0]), int((1-overlap)*self.crop_size[0]))
                offset_x = np.random.randint(int(overlap*self.crop_size[1]), int((1-overlap)*self.crop_size[1]))
                crop_y1 = np.clip(sel_pix_y - offset_y, 0, img.shape[0] - self.crop_size[0])
                crop_y2 = crop_y1 + self.crop_size[0]
                crop_x1 = np.clip(sel_pix_x - offset_x, 0, img.shape[1] - self.crop_size[1])
                crop_x2 = crop_x1 + self.crop_size[1]
            else:
                offset_h = np.random.randint(0, max(img.shape[0] - self.crop_size[0], 0))
                offset_w = np.random.randint(0, max(img.shape[1] - self.crop_size[1], 0))
                crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
                crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img, results['gt_seg_map'])
        # if self.cat_max_ratio < 1.:
        #     # Repeat 10 times
        #     for _ in range(10):
        #         seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
        #         labels, cnt = np.unique(seg_temp, return_counts=True)
        #         cnt = cnt[labels != self.ignore_index]
        #         if len(cnt) > 1 and np.max(cnt) / np.sum(
        #                 cnt) < self.cat_max_ratio:
        #             break
        #         crop_bbox = generate_crop_bbox(img, results['gt_seg_map'])

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
