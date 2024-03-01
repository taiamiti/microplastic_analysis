import argparse
import glob
import os

from tqdm import tqdm

from mmseg.apis import MMSegInferencer
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model_cfg', help='train config file path')
    parser.add_argument('--model_ckpts', help='model ckpt file')
    parser.add_argument("--img_folder", help="the dir to look for input images")
    parser.add_argument("--save_folder", help="the dir to look for images")
    parser.add_argument("--thresh", default=0.5, help="Segmentation threshold")
    args = parser.parse_args()
    return args


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(model_cfg, model_cktps, img_folder, save_folder, thresh=0.5):
    """

    Args:
        model_cfg:
        model_cktps:
        img_folder:
        save_folder:
        thresh:

    Returns:

    Example :
    model_cfg = "projects/microplastic_detection/configs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test.py
    model_cktps = "./work_dirs/fcn-unet-s5-d16_unet_1xb16-0.0001-20k_microplastic_detection-400x400_train_test/best_mIoU_iter_2800.pth"
    img_folder = "data/processed/create_composite/lot11-20-11-2023-eau/data"
    save_folder = "pred/lot11-20-11-2023-eau"
    """
    img_paths = get_img_path_from_dir(img_folder)
    classes = ['background', 'microplastic']
    palette = [[255, 0, 0], [0, 0, 255]]
    engine = MMSegInferencer(model=model_cfg, weights=model_cktps, classes=classes, palette=palette)
    for img_path in tqdm(img_paths):
        results = engine(os.path.join(img_folder, img_path), return_datasamples=True)
        logits = results.seg_logits.data.cpu().numpy()
        prob = sigmoid(logits[1, :, :])
        seg = (prob > thresh).astype(np.uint8)
        pilimg = Image.fromarray(255*seg)
        save_path = os.path.join(save_folder, img_path.replace(".jpg", ".png"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pilimg.save(save_path)


def get_img_paths_from_protocol(protocol):
    with open(protocol, 'r') as fp:
        img_paths = fp.readlines()
    return [img_path.strip('\n') for img_path in img_paths]


def get_img_path_from_dir(dir_path):
    return [os.path.relpath(x, dir_path) for x in glob.glob(os.path.join(dir_path, "*.jpg"))]


if __name__ == "__main__":
    args = parse_args()
    main(args.model_cfg, args.model_ckpts, args.img_folder, args.save_folder, args.thresh)

