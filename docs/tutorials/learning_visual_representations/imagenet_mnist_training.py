import os
import yaml
import argparse
import pprint

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms.v2 as T

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import ImageNetMnist, collate_views
from learning.transformations import DataTransformation
from learning.nce_loss import nce_loss
from utils import deprocess_image

from models.alexnet import alexnet
from models.projection_head import ProjectionHead


def get_shared_feature_key(shared_feature):
    """Return a filesystem-safe string key for the shared_feature."""
    if isinstance(shared_feature, str):
        return shared_feature
    elif isinstance(shared_feature, list):
        return '_'.join(sorted(shared_feature))
    return str(shared_feature)


def main(cfg):
    shared_feature = cfg['shared_feature']
    shared_feature_key = get_shared_feature_key(shared_feature)

    # Per-experiment output directory
    exp_dir = os.path.join(cfg['res_dir'], shared_feature_key)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'train_config.yaml'), 'w') as file:
        yaml.dump(cfg, file)

    # Support both 'data_folder'/'labels_file' (config) and
    # 'imagenet_data_folder'/'imagenet_labels_file' (legacy) key names
    data_folder = cfg.get('imagenet_data_folder', cfg.get('data_folder'))
    labels_file = cfg.get('imagenet_labels_file', cfg.get('labels_file'))

    dataset = ImageNetMnist(
        imagenet_data_folder=data_folder,
        imagenet_labels_file=labels_file,
        imagenet_classes=cfg['imagenet_classes'],
        mnist_data_folder=cfg['mnist_data_folder'],
        shared_feature=shared_feature)

    transform = DataTransformation(cfg)
    if shared_feature == 'background' or (isinstance(shared_feature, list) and 'background' in shared_feature):
        dataset.transform1 = transform(['random_cropping', 'resize'])
        dataset.transform2 = transform(['gaussian_blur', 'normalize'])
    elif shared_feature == 'digit':
        dataset.transform1 = transform(['center_cropping'])
        dataset.transform2 = transform(['normalize'])
    else:
        raise ValueError("Shared feature must be 'background', 'digit', or a list containing both.")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_views
    )

    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = alexnet(out=cfg['d_model'], sobel=True, freeze_features=False)

    pretrained_params_path = cfg.get('pretrained_params')
    if pretrained_params_path is not None:
        pretrained_params = torch.load(pretrained_params_path, map_location=device)
        # Replace top_layer weights to match d_model output size
        pretrained_params['top_layer.weight'] = torch.randn(
            (cfg['d_model'], pretrained_params['top_layer.weight'].shape[1])
        )
        pretrained_params['top_layer.bias'] = torch.randn(cfg['d_model'])
        model.load_state_dict(pretrained_params)

    model = model.to(device)

    projection_head = ProjectionHead(d_in=cfg['d_model'], d_model=cfg['d_model']).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(projection_head.parameters()),
        lr=float(cfg['lr'])
    )

    metrics = {'train': {'loss': []}}

    best_train_loss = np.inf
    model.train()
    for epoch in range(cfg['epochs']):
        train_loss = 0
        model.train()
        for imgs, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            img1 = imgs['view1'].to(device)
            img2 = imgs['view2'].to(device)

            h1 = model(img1)
            h2 = model(img2)

            z1 = projection_head(h1)
            z2 = projection_head(h2)

            loss = nce_loss(z1, z2, temperature=cfg['temperature'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(data_loader)

        print("Train metrics - Loss: {:.4f}".format(train_loss))
        metrics['train']['loss'].append(train_loss)

        if train_loss <= best_train_loss:
            best_train_loss = train_loss
            torch.save(
                {'epoch': epoch + 1,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()},
                os.path.join(exp_dir, 'best_model.pth.tar')
            )

    # ── Save images and labels for KNN evaluation ─────────────────────────
    print("Saving images and labels for KNN evaluation...")

    eval_dataset = ImageNetMnist(
        imagenet_data_folder=data_folder,
        imagenet_labels_file=labels_file,
        imagenet_classes=cfg['imagenet_classes'],
        mnist_data_folder=cfg['mnist_data_folder'],
        shared_feature=shared_feature)

    # Consistent 224×224 center-cropped + normalized images for all configs
    eval_dataset.transform1 = T.Compose([T.Resize(256), T.CenterCrop(224)])
    eval_dataset.transform2 = T.Normalize(mean=cfg['data_mean'], std=cfg['data_std'])

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_views
    )

    all_images, all_imagenet_labels, all_digit_labels = [], [], []
    for imgs, labels in tqdm(eval_loader, desc="Collecting eval data"):
        all_images.append(imgs['view1'])
        all_imagenet_labels.append(labels['view1']['imagenet_label'])
        all_digit_labels.append(labels['view1']['digit_label'])

    all_images = torch.cat(all_images)
    all_imagenet_labels = torch.cat(all_imagenet_labels)
    all_digit_labels = torch.cat(all_digit_labels)

    torch.save(all_images,          os.path.join(exp_dir, f'test_data_{shared_feature_key}.pt'))
    torch.save(all_imagenet_labels, os.path.join(exp_dir, f'test_imagenet_labels_{shared_feature_key}.pt'))
    torch.save(all_digit_labels,    os.path.join(exp_dir, f'test_digit_labels_{shared_feature_key}.pt'))
    print(f"Saved eval data to {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        help='Path to the training config.')

    parser = parser.parse_args()

    with open(parser.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    pprint.pprint(cfg)
    main(cfg)