import json

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


def compute_image_embeddings(model, preprocess, fpath):
    image = preprocess(Image.open(fpath)).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def compute_embeddings(image_paths):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    features_list = []
    for image_path in tqdm(image_paths):
        features = compute_image_embeddings(model, preprocess, image_path)
        features_list.append(features)
    # Compute embeddings
    embeddings = torch.cat(features_list).cpu().detach().numpy()
    return embeddings


def load_embedding_centers(fpath):
    with open(fpath, "r") as fp:
        embedding_centers_serialized = json.load(fp)

    center_label = list(embedding_centers_serialized.keys())
    embedding_centers = np.array(list(embedding_centers_serialized.values()))
    return embedding_centers, center_label
