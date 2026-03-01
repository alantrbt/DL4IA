import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision.transforms.v2 as T

import numpy as np

from utils import read_image

from typing import Tuple, Iterable, Union


class ImageNet(Dataset):
    """TODO: complete.
    """
    def __init__(self, data_folder: str, labels_file: str):
        super().__init__()
        self.data_folder = data_folder
        self.files = os.listdir(data_folder)
        self.files.sort()

        with open(labels_file, 'r') as f:
            labels = f.readlines()
        self.labels = [int(x.split('\n')[0]) for x in labels]
        
        self.transform = None

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file_path = os.path.join(self.data_folder, self.files[i])
        img = read_image(file_path)

        # Convert to tensor
        img = torch.from_numpy(img)  # TODO: complete.

        # # Permute axis
        img = img.permute(2, 0, 1)
        # print(f"Image shape after permutation: {img.shape}")  # Debug print

        # Convert to float in [0, 1]
        img = img.float() / 255.0  # TODO: complete.

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[i] - 1
        return img, label


class SubsetImageNet(ImageNet):
    """A subset of ImageNet given a subset of classes, e.g. [1, 2, 3].
    """
    def __init__(self, data_folder: str, labels_file: str, classes: list):
        super().__init__(data_folder, labels_file)

        self.labels = torch.tensor(self.labels)
        self.mask = torch.zeros_like(self.labels)
        self.classes = classes

        for class_id in classes:
            self.mask[self.labels == class_id] = 1

        self.indices = torch.where(self.mask == 1)[0]

    def __len__(self):
        return self.mask.sum()
    
    def __getitem__(self, i):
        img, label = super().__getitem__(self.indices[i])
        label = self.classes.index(label.item() + 1)
        return img, label
    

class ContrastiveDataset(SubsetImageNet):
    """A dataset for contrastive learning: 
    outputs two random views of the same data.
    """
    def __init__(self, data_folder: str, labels_file: str, classes: list, 
                 transform: T.Transform):
        super().__init__(data_folder, labels_file, classes)
        self.view_transform = transform

    def __getitem__(self, i):
        img, _ = super().__getitem__(i)
        img1 = self.view_transform(img)  # TODO: complete.
        img2 = self.view_transform(img)  # TODO: complete.
        return img, img1, img2


class ImageNetMnist(SubsetImageNet):
    """A simplified version of Colorful-Moving-Mnist:
    https://proceedings.neurips.cc/paper_files/paper/2020/file/4c2e5eaae9152079b9e95845750bb9ab-Paper.pdf
    Provides ImageNet images with either 1 or 7 overlapped digits.
    """
    def __init__(
            self, 
            imagenet_data_folder: str, 
            imagenet_labels_file: str, 
            imagenet_classes: list, 
            mnist_data_folder: str,
            shared_feature: Union[str, list]):
        super().__init__(imagenet_data_folder, imagenet_labels_file, imagenet_classes)

        self.mnist_data = np.load(os.path.join(mnist_data_folder, 'mnist_subset_data.npy'))
        self.mnist_labels = np.load(os.path.join(mnist_data_folder, 'mnist_subset_labels.npy'))
        self.digit_size = int(self.mnist_data.shape[-1]**0.5)
        self.shared_feature = shared_feature
        self.transform1 = lambda x: x
        self.transform2 = lambda x: x


    def get_digit(self, i):
        digit = self.mnist_data[i].reshape(self.digit_size, self.digit_size)
        digit = torch.from_numpy(digit).float() / 255
        digit = digit.unsqueeze(0).repeat(3, 1, 1)
        label = self.mnist_labels[i]
        return digit, label
    
    def __getitem__(self, i):
        img, imagenet_label1 = super().__getitem__(i)
        random_digit_ind = np.random.randint(len(self.mnist_labels))
        digit1, digit_label1 = self.get_digit(random_digit_ind)

        img1 = self.transform1(img)     
        img1 = insert_digit(digit1, img1)

        if isinstance(self.shared_feature, str):
            if self.shared_feature == 'background':
                random_digit_ind2 = np.where(self.mnist_labels != digit_label1)
                random_digit_ind2 = random_digit_ind2[0][np.random.randint(len(random_digit_ind2[0]))]
                digit2, digit_label2 = self.get_digit(random_digit_ind2)

                img2 = self.transform1(img)
                img2 = insert_digit(digit2, img2)
                imagenet_label2 = imagenet_label1

            elif self.shared_feature == 'digit':
                random_ind = np.random.choice(
                    np.concatenate((
                        np.arange(i),
                        np.arange(i + 1, len(self))
                    )), size=1
                )
                
                img2, imagenet_label2 = super().__getitem__(random_ind[0])
                img2 = self.transform1(img2)
                img2 = insert_digit(digit1, img2)
                digit_label2 = digit_label1
            
            else:
                raise ValueError("If shared_feature is a string,\
                                then the shared feature must either be 'background' or 'digit'")
        elif isinstance(self.shared_feature, list):
            if 'background' in self.shared_feature and 'digit' in self.shared_feature:
                ...
            else:
                raise ValueError("If shared_feature is a list,\
                                then it must contain 'background' and 'digit'.")

        imgs = {
            'original': img,
            'view1': self.transform2(img1),
            'view2': self.transform2(img2)
        }

        labels = {
            'view1': {
                'imagenet_label': imagenet_label1,
                'digit_label': int(digit_label1),
            },
            'view2': {
                'imagenet_label': imagenet_label2,
                'digit_label': int(digit_label2),
            }
        }
        return imgs, labels
    

def insert_digit(digit: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    resize = T.Resize(img.shape[-1])
    digit = resize(digit)
    image = torch.clone(img)
    image += digit
    image = torch.clamp(image, min=0, max=1)
    return image


def collate_views(batch: Iterable[dict[str, torch.Tensor]]) -> Tuple[dict[str, torch.Tensor], ...]:
    """ Collate function (draft generated by Le Chat, an AI assistant developed by Mistral AI).
    Fun fact: when I asked to Le Chat if my type hint was Pythonic, it answered: 
    'Iterable is correct, but it's often more Pythonic to use Sequence or Iterable'!
    For reliable information about typing: https://docs.python.org/3/library/typing.html#
    """
    batched_imgs = {
        'original': [],
        'view1': [],
        'view2': []
    }
    batched_labels = {
        'view1': {
            'imagenet_label': [],
            'digit_label': [],
        },
        'view2': {
            'imagenet_label': [],
            'digit_label': [],
        }
    }

    # Collect images and labels from each sample in the batch
    for imgs, labels in batch:
        batched_imgs['original'].append(imgs['original'])
        batched_imgs['view1'].append(imgs['view1'])
        batched_imgs['view2'].append(imgs['view2'])

        batched_labels['view1']['imagenet_label'].append(labels['view1']['imagenet_label'])
        batched_labels['view1']['digit_label'].append(labels['view1']['digit_label'])

        batched_labels['view2']['imagenet_label'].append(labels['view2']['imagenet_label'])
        batched_labels['view2']['digit_label'].append(labels['view2']['digit_label'])

    # Stack images into tensors
    batched_imgs = {
        'original': torch.stack(batched_imgs['original'], dim=0),
        'view1': torch.stack(batched_imgs['view1'], dim=0),
        'view2': torch.stack(batched_imgs['view2'], dim=0)
    }

    # Stack labels into tensors
    batched_labels = {
        'view1': {
            'imagenet_label': torch.tensor(batched_labels['view1']['imagenet_label']),
            'digit_label': torch.tensor(batched_labels['view1']['digit_label']),
        },
        'view2': {
            'imagenet_label': torch.tensor(batched_labels['view2']['imagenet_label']),
            'digit_label': torch.tensor(batched_labels['view2']['digit_label']),
        }
    }

    return batched_imgs, batched_labels