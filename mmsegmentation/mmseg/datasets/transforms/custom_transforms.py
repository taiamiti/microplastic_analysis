from typing import Union, Tuple

import numpy as np
from mmcv.transforms import BaseTransform, TRANSFORMS
from mmcv.transforms.utils import cache_randomness
import cv2


@TRANSFORMS.register_module()
class RandomCropChoice(BaseTransform):
    """Randomly choose between RandomCrop, RandomCropForeground, and CornerCrop.

    This transform allows mixing standard random cropping, foreground-focused
    cropping, and corner-based cropping with configurable probabilities.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map

    Args:
        crop_size (Union[int, Tuple[int, int]]): Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        foreground_prob (float): Probability of using RandomCropForeground.
            Default: 0.8
        corner_prob (float): Probability of using CornerCrop.
            Default: 0.0
        Note: random_prob = 1.0 - foreground_prob - corner_prob
        cat_max_ratio (float): The maximum ratio that single category could
            occupy. Default: 1.0
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 foreground_prob: float = 0.8,
                 corner_prob: float = 0.0,
                 cat_max_ratio: float = 1.0,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two integers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        assert 0.0 <= foreground_prob <= 1.0, 'foreground_prob must be between 0 and 1'
        assert 0.0 <= corner_prob <= 1.0, 'corner_prob must be between 0 and 1'
        assert foreground_prob + corner_prob <= 1.0, \
            'sum of foreground_prob and corner_prob must not exceed 1.0'

        self.crop_size = crop_size
        self.foreground_prob = foreground_prob
        self.corner_prob = corner_prob
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def get_crop_mode(self) -> str:
        """Randomly decide which crop mode to use.

        Returns:
            str: One of 'foreground', 'corner', or 'random'.
        """
        rand_val = np.random.rand()

        if rand_val < self.foreground_prob:
            return 'foreground'
        elif rand_val < self.foreground_prob + self.corner_prob:
            return 'corner'
        else:
            return 'random'

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """Get a crop bounding box based on the selected mode.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """
        img = results['img']
        gt_seg_map = results['gt_seg_map']

        crop_mode = self.get_crop_mode()

        if crop_mode == 'foreground':
            # Use foreground-focused cropping
            return self._generate_foreground_crop_bbox(img, gt_seg_map)
        elif crop_mode == 'corner':
            # Use corner cropping
            return self._generate_corner_crop_bbox(img)
        else:
            # Use standard random cropping
            return self._generate_random_crop_bbox(img)

    def _generate_foreground_crop_bbox(self, img: np.ndarray, gt_seg_map: np.ndarray) -> tuple:
        """Generate a crop bbox centered on foreground pixels.

        Args:
            img (np.ndarray): Original input image.
            gt_seg_map (np.ndarray): Ground truth segmentation map.

        Returns:
            tuple: Coordinates of the cropped image.
        """
        overlap = 0.25
        yis, xis = np.where(gt_seg_map == 1)

        if len(yis) > 0 and len(xis) > 0:
            # Pick a random foreground pixel
            sel_pix_y = yis[np.random.randint(0, yis.shape[0])]
            sel_pix_x = xis[np.random.randint(0, xis.shape[0])]

            # Random offset to ensure pixel is within crop
            offset_y = np.random.randint(
                int(overlap * self.crop_size[0]),
                int((1 - overlap) * self.crop_size[0])
            )
            offset_x = np.random.randint(
                int(overlap * self.crop_size[1]),
                int((1 - overlap) * self.crop_size[1])
            )

            crop_y1 = np.clip(sel_pix_y - offset_y, 0, img.shape[0] - self.crop_size[0])
            crop_y2 = crop_y1 + self.crop_size[0]
            crop_x1 = np.clip(sel_pix_x - offset_x, 0, img.shape[1] - self.crop_size[1])
            crop_x2 = crop_x1 + self.crop_size[1]
        else:
            # No foreground pixels, fall back to random crop
            return self._generate_random_crop_bbox(img)

        return crop_y1, crop_y2, crop_x1, crop_x2

    def _generate_corner_crop_bbox(self, img: np.ndarray) -> tuple:
        """Generate a crop bbox from one of the four corners.

        Args:
            img (np.ndarray): Original input image.

        Returns:
            tuple: Coordinates of the cropped image.
        """
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)

        # Randomly select one of the four corners
        corner = np.random.randint(0, 4)

        if corner == 0:  # Top-left
            crop_y1, crop_y2 = 0, self.crop_size[0]
            crop_x1, crop_x2 = 0, self.crop_size[1]
        elif corner == 1:  # Top-right
            crop_y1, crop_y2 = 0, self.crop_size[0]
            crop_x1, crop_x2 = margin_w, margin_w + self.crop_size[1]
        elif corner == 2:  # Bottom-left
            crop_y1, crop_y2 = margin_h, margin_h + self.crop_size[0]
            crop_x1, crop_x2 = 0, self.crop_size[1]
        else:  # Bottom-right (corner == 3)
            crop_y1, crop_y2 = margin_h, margin_h + self.crop_size[0]
            crop_x1, crop_x2 = margin_w, margin_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def _generate_random_crop_bbox(self, img: np.ndarray) -> tuple:
        """Generate a standard random crop bbox.

        Args:
            img (np.ndarray): Original input image.

        Returns:
            tuple: Coordinates of the cropped image.
        """
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)

        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from image.

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
        """Transform function to randomly crop images and segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # Crop the image
        img = self.crop(img, crop_bbox)

        # Crop semantic segmentation
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'crop_size={self.crop_size}, '
                f'foreground_prob={self.foreground_prob}, '
                f'corner_prob={self.corner_prob})')


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
