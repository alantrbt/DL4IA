import os
import yaml
import argparse
import pprint

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as T

from datasets import ContrastiveDataset
from models.alexnet import alexnet
from models.projection_head import ProjectionHead
from learning.nce_loss import nce_loss
from learning.transformations import DataTransformation

from tqdm import tqdm


def main(cfg):
    os.makedirs(os.path.dirname(cfg['res_dir']), exist_ok=True)

    with open(os.path.join(cfg['res_dir'], 'train_config.yaml'), 'w') as file:
        yaml.dump(cfg, file)

    subset_classes = cfg['subset_classes']

    transform = DataTransformation(cfg)

    dataset = ContrastiveDataset(
        cfg['data_folder'], 
        cfg['labels_file'], 
        classes=subset_classes,
        transform=transform()
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True
    )

    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = alexnet(out=cfg['d_alexnet'], sobel=True, freeze_features=False)
    pretrained_params = torch.load(cfg['pretrained_params'])
    model.load_state_dict(pretrained_params)
    model = model.to(device)

    projection_head = ProjectionHead(d_in=cfg['d_alexnet'], d_model=cfg['d_model']).to(device)

    optimizer = optim.Adam(
            list(model.parameters()) + list(projection_head.parameters()),
            lr=float(cfg['lr'])
    )

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    
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

    best_train_loss = np.inf
    model.train()
    for epoch in range(cfg['epochs']):
        train_loss = 0
        model.train()
        for _, img1, img2 in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            img1 = img1.to(device)
            img2 = img2.to(device)

            h1, h2 = model(img1), model(img2)
            z1, z2 = projection_head(h1), projection_head(h2)

            loss = nce_loss(z1, z2, temperature=cfg['temperature'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(data_loader)        

        print("Train metrics - Loss: {:.4f}".format(train_loss))
        metrics['train']['loss'].append(train_loss)

        if train_loss <= best_train_loss:
            best_train_loss = train_loss
            torch.save({'epoch': epoch + 1, 
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(cfg['res_dir'], 'best_model.pth.tar'))
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        help='Path to the training config.')
   
    parser = parser.parse_args()

    with open(parser.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    pprint.pprint(cfg)
    main(cfg)