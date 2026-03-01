import os
import yaml
import argparse
import pprint

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, Subset

import torchvision.transforms.v2 as T

from datasets import SubsetImageNet
from models.alexnet import alexnet
from tqdm import tqdm


def main(cfg):
    os.makedirs(os.path.dirname(cfg['res_dir']), exist_ok=True)

    with open(os.path.join(cfg['res_dir'], 'train_config.yaml'), 'w') as file:
        yaml.dump(cfg, file)

    subset_classes = cfg['subset_classes']

    dataset = SubsetImageNet(
        cfg['data_folder'], 
        cfg['labels_file'], 
        classes=subset_classes
        )
    
    transformations = {
        'center_crop': T.CenterCrop(cfg['crop_size']),
        'normalize': T.Normalize(
            mean=cfg['data_mean'],
            std=cfg['data_std'])
    }

    dataset.transform = T.Compose(
        [transformations[k] for k in cfg['transforms']]
        )

    n_samples = len(dataset)
    n_train_samples = int(n_samples * 0.6)
    train_dataset = Subset(dataset, np.arange(n_train_samples))
    test_dataset = Subset(dataset, np.arange(n_train_samples, n_samples))

    train_dataset, val_dataset = random_split(dataset, [1 - cfg['val_split'], cfg['val_split']])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
    )

    val_test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
    )

    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = alexnet(out=cfg['n_classes'], sobel=True, freeze_features=True).to(device)

    for name, param in model.named_parameters():
        if 'features' in name or 'sobel' in name:
            assert param.requires_grad == False, "Feature layers should be frozen."
        else:
            assert param.requires_grad == True, "Only feature layers should be frozen."


    optimizer = optim.Adam(model.parameters(), lr=float(cfg['lr']))

    metrics = {
            'train': {
                'loss': [],
                'acc': []
            },
            'val': {
                'loss': [],
                'acc': []
            }
        }

    best_val_loss = np.inf
    best_val_acc = 0

    for epoch in range(cfg['epochs']):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for data, labels in tqdm(train_data_loader, desc=f"Epoch {epoch + 1}"):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)
            accuracy = (pred == labels).float().mean().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_data_loader)
            train_acc += accuracy / len(train_data_loader)

        print("Train metrics - Loss: {:.4f} - Acc: {:.2f}".format(train_loss, train_acc))
        metrics['train']['loss'].append(train_loss)
        metrics['train']['acc'].append(train_acc)

        model.eval()
        for data, labels in tqdm(val_data_loader, desc=f"Validation"):
            data = data.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(data)

            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)
            accuracy = (pred == labels).float().mean().item()

            val_loss += loss.item() / len(val_data_loader)
            val_acc += accuracy / len(val_data_loader)

        print("Val metrics - Loss: {:.4f} - Acc: {:.2f}".format(val_loss, val_acc))
        metrics['val']['loss'].append(val_loss)
        metrics['val']['acc'].append(val_acc)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(cfg['res_dir'], 'best_model.pth.tar'))
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    print("Best validation loss: {:.2f}".format(val_loss))
    print("Best validation accuracy: {:.2f}".format(val_acc))

    ...
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        help='Path to the training config.')
   
    parser = parser.parse_args()

    with open(parser.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    pprint.pprint(cfg)
    main(cfg)
