import torchvision.transforms.v2 as T


# class DataTransformation:
#     """Compose data transformations. TODO: complete.
#     """
#     def __init__(self, cfg):
#         self.transforms = cfg['transforms']
#         transformations = {}

#         if 'center_cropping' in cfg['transforms']:
#             assert 'crop_size' in cfg
#             transformations['center_cropping'] = ...

#         if 'random_cropping' in cfg['transforms']:
#             assert 'crop_size' in cfg
#             transformations['random_cropping'] = ...

#         if 'resize' in cfg['transforms']:
#             assert 'img_size' in cfg
#             transformations['resize'] = ...

#         if 'color_distortion' in cfg['transforms']:
#             assert 'brightness_range' in cfg
#             assert 'contrast_range' in cfg
#             assert 'saturation_range' in cfg
#             assert 'hue_range' in cfg
#             transformations['color_distortion'] = ...

#         if 'gaussian_blur' in cfg['transforms']:
#             assert 'kernel_size' in cfg
#             transformations['gaussian_blur'] = ...

#         if 'normalize' in cfg['transforms']:
#             assert 'data_mean' in cfg
#             assert 'data_std' in cfg
#             transformations['normalize'] = ...

#         self.transformations = transformations

#     def __call__(self, transforms=None):
#         if transforms is None:
#             return T.Compose([self.transformations[k] for k in self.transformations])
#         else:


class DataTransformation:
    """Compose data transformations. TODO: complete.
    """
    def __init__(self, cfg):
        self.transforms = cfg['transforms']
        transformations = {}

        if 'center_cropping' in cfg['transforms']:
            assert 'crop_size' in cfg
            transformations['center_cropping'] = T.CenterCrop(cfg['crop_size'])

        if 'random_cropping' in cfg['transforms']:
            assert 'crop_size' in cfg
            transformations['random_cropping'] = T.RandomCrop(cfg['crop_size'])

        if 'resize' in cfg['transforms']:
            assert 'img_size' in cfg
            transformations['resize'] = T.Resize(cfg['img_size'])

        if 'color_distortion' in cfg['transforms']:
            assert 'brightness_range' in cfg
            assert 'contrast_range' in cfg
            assert 'saturation_range' in cfg
            assert 'hue_range' in cfg
            transformations['color_distortion'] = T.ColorJitter(
                brightness=cfg['brightness_range'],
                contrast=cfg['contrast_range'],
                saturation=cfg['saturation_range'],
                hue=cfg['hue_range']
            )

        if 'gaussian_blur' in cfg['transforms']:
            assert 'kernel_size' in cfg
            transformations['gaussian_blur'] = T.GaussianBlur(cfg['kernel_size'])

        if 'normalize' in cfg['transforms']:
            assert 'data_mean' in cfg
            assert 'data_std' in cfg
            transformations['normalize'] = T.Normalize(mean=cfg['data_mean'], std=cfg['data_std'])

        self.transformations = transformations

    def __call__(self, transforms=None):
        if transforms is None:
            return T.Compose([self.transformations[k] for k in self.transformations])
        else:
            return T.Compose([self.transformations[k] for k in transforms])