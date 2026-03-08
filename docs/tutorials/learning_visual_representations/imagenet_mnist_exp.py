import os
import yaml
import argparse
import pprint

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier

from models.alexnet import alexnet

import pdb


def get_shared_feature_key(shared_feature):
    """Return a filesystem-safe string key for the shared_feature."""
    if isinstance(shared_feature, str):
        return shared_feature
    elif isinstance(shared_feature, list):
        return '_'.join(sorted(shared_feature))
    return str(shared_feature)


def main(cfg, task: str = 'imagenet'):
    shared_feature = cfg['shared_feature']
    shared_feature_key = get_shared_feature_key(shared_feature)
    exp_folder = os.path.join(cfg['res_dir'], shared_feature_key)

    data = torch.load(os.path.join(exp_folder, f'test_data_{shared_feature_key}.pt'))
    labels = torch.load(os.path.join(exp_folder, f'test_{task}_labels_{shared_feature_key}.pt'))

    n = len(labels) // 2
    train_data, test_data = data[:n], data[n:]
    train_labels, test_labels = labels[:n], labels[n:]

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
    )

    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = alexnet(out=cfg['d_model'], sobel=True)
    pretrained_params = torch.load(os.path.join(exp_folder, 'best_model.pth.tar'))
    model.load_state_dict(pretrained_params['model'])
    model = model.to(device)
    model.eval()

    train_features = []
    for imgs, labels in tqdm(train_data_loader, "Compute representations on the train set..."):
        imgs = imgs.to(device)
        with torch.no_grad():
            h = model(imgs)
        train_features.append(h)

    test_features = []
    for imgs, labels in tqdm(test_data_loader, "Compute representations on the test set..."):
        imgs = imgs.to(device)
        with torch.no_grad():
            h = model(imgs)
        test_features.append(h)

    train_features = torch.cat(train_features).cpu().numpy()
    test_features = torch.cat(test_features).cpu().numpy()
    train_labels, test_labels = train_labels.numpy(), test_labels.numpy()

    knn = KNeighborsClassifier(n_neighbors=cfg.get('n_neighbors', 5))
    knn.fit(train_features, train_labels)
    pred = knn.predict(test_features)

    test_acc = np.mean(pred == test_labels)

    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        help='Path to the training config.')

    parser = parser.parse_args()

    with open(parser.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    pprint.pprint(cfg)

    shared_feature_key = get_shared_feature_key(cfg['shared_feature'])
    exp_folder = os.path.join(cfg['res_dir'], shared_feature_key)

    metrics = {}
    for task in ['imagenet', 'digit']:
        acc = main(cfg, task)
        metrics[task] = float(acc)

    with open(os.path.join(exp_folder, f'test_accuracy_{shared_feature_key}.yaml'), 'w') as file:
        yaml.dump(metrics, file)
