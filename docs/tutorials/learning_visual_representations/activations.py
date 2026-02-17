import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter

from tqdm import tqdm

from utils import deprocess_image

def compute_batch_activations(model, x, layer):
    """TODO: complete.
    """
    if model.sobel is not None:
        x = model.sobel(x)
    current_layer = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = ...
            if isinstance(m, nn.ReLU):
                if current_layer == layer:
                    activation = ...
                else:
                    ...


def compute_activations_for_gradient_ascent(model, x, layer, filter_id):
    """TODO: complete.
    """
    if model.sobel is not None:
        x = model.sobel(x)
    current_layer = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = ...
            if isinstance(m, nn.Conv2d) and current_layer == layer:
                activation = ...
            if isinstance(m, nn.ReLU):
                if current_layer == layer:
                    if x[:, filter_id, :, :].mean().data.cpu().numpy() == 0:
                        return activation
                    else:
                        activation = ...
                    return activation
                else:
                    ...


def compute_dataset_activations(model, dataset, layer, batch_size=64):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size
    )

    activations = []
    for data, _ in tqdm(loader, desc=f"Compute activations over dataset for layer {layer}"):
        batch_activation = compute_batch_activations(model, data, layer)
        activations.append(batch_activation)

    return torch.cat(activations)


def maximize_img_response(model, img_size, layer, filter_id, device='cuda', n_it=50000, wd=1e-5, lr=3, reg_step=5):
    """TODO: complete.
    A L2 regularization is combined with a Gaussian blur operator applied every reg_step steps.
    """
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for param in model.parameters():
        param.requires_grad_(False)

    img = torch.nn.Parameter(
        data=torch.randn((1, 3, img_size, img_size))
        ).to(device)
    
    model = model.to(device)
    for it in tqdm(range(n_it), desc='Gradient ascent in image space'):

        out = compute_activations_for_gradient_ascent(
            model, img, layer=layer, filter_id=filter_id
            )
        target = torch.autograd.Variable(...).to(device)
        loss = F.cross_entropy(out, target) + wd * ...

        # compute gradient
        ...

        # normalize gradient
        grads = img.grad.data
        grads = grads.div(torch.norm(grads)+1e-8)

        # Update image
        ...

        # Apply gaussian blur
        if it % reg_step == 0:
            img = gaussian_filter(torch.squeeze(img).detach().cpu().numpy().transpose((2, 1, 0)),
                                    sigma=(0.3, 0.3, 0))
            img = torch.unsqueeze(torch.from_numpy(img).float().transpose(2, 0), 0)
            img = torch.nn.Parameter(data=img).to(device)

    return deprocess_image(img.detach().cpu().numpy())