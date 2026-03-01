import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter

from tqdm import tqdm

from utils import deprocess_image


def compute_batch_activations(model, x, layer):
    """TODO: complete.
    """
    activation = None  # TODO: complete.
    if model.sobel is not None:
        x = model.sobel(x)
    current_layer = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)  # TODO: complete.
            if isinstance(m, nn.ReLU):
                if current_layer == layer:
                    activation = x.clone().detach()  # TODO: complete.
                else:
                    current_layer += 1  # TODO: complete.
    if activation is None:  # TODO: complete.
        raise ValueError(f"Layer {layer} not found in model.")  # TODO: complete.
    return activation  # TODO: complete.


def compute_activations_for_gradient_ascent(model, x, layer, filter_id):
    """TODO: complete.
    """
    activation = None  # TODO: complete.
    if model.sobel is not None:
        x = model.sobel(x)
    current_layer = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)  # TODO: complete.
            if isinstance(m, nn.Conv2d) and current_layer == layer:
                activation = x #.clone()#.detach()  # TODO: complete.
            if isinstance(m, nn.ReLU):
                if current_layer == layer:
                    filter_activation = x[:, filter_id, :, :]
                    if filter_activation.mean().item() == 0:
                        return activation
                    else:
                        activation = filter_activation #.clone().detach()  # TODO: complete.
                    return activation
                else:
                    current_layer += 1  # TODO: complete.
    if activation is None:  # TODO: complete.
        raise ValueError(f"Layer {layer} not found in model.")  # TODO: complete.


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


# def maximize_img_response(model, img_size, layer, filter_id, device='cuda', n_it=50000, wd=1e-5, lr=3, reg_step=5):
#     """TODO: complete.
#     A L2 regularization is combined with a Gaussian blur operator applied every reg_step steps.
#     """
#     if device == 'cuda' and torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'

#     for param in model.parameters():
#         param.requires_grad_(False)

#     # img = torch.nn.Parameter(
#     #     data=torch.randn((1, 3, img_size, img_size))
#     #     ).to(device)
    
#     img = torch.randn(
#     (1, 3, img_size, img_size),
#     device=device,
#     requires_grad=True
# )
    
#     model = model.to(device)
#     for it in tqdm(range(n_it), desc='Gradient ascent in image space'):

#         out = compute_activations_for_gradient_ascent(
#             model, img, layer=layer, filter_id=filter_id
#             )
#         print(out.shape)
#         print(out.dtype)
#         target = torch.zeros(out.size(0), dtype=torch.long).to(device)
#         print(target.shape)
#         print(target.dtype)
#         print(type(img))
#         print(1)
#         loss = - out.mean() + wd * torch.norm(img)  #done# TODO: complete.
#         print("---------------")
#         print(loss.item())
#         print(out.requires_grad)
#         print(loss.requires_grad)
#         print("requires_grad:", img.requires_grad)
#         print("is_leaf:", img.is_leaf)
#         print("grad_fn:", img.grad_fn)

#         # compute gradient
#         loss.backward()  # TODO: complete.
#         print("backward pass completed")

#         # normalize gradient
#         print(type(img.grad))
#         print(2)
#         grads = img.grad.data  # TODO: complete.
#         grads = grads.div(torch.norm(grads)+1e-8)

#         # Update image
#         img.data = img.data + lr * grads  # TODO: complete.
#         img.grad.zero_()  # TODO: complete.

#         # Apply gaussian blur
#         if it % reg_step == 0:
#             img = gaussian_filter(torch.squeeze(img).detach().cpu().numpy().transpose((2, 1, 0)),
#                                     sigma=(0.3, 0.3, 0))
#             img = torch.unsqueeze(torch.from_numpy(img).float().transpose(2, 0), 0)
#             img = torch.nn.Parameter(data=img).to(device)

#     return deprocess_image(img.detach().cpu().numpy())

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

    # CORRECTION : Créer le tenseur directement sur le device AVANT d'en faire un Parameter
    # Cela garantit que img.grad ne sera pas None.
    initial_tensor = torch.randn((1, 3, img_size, img_size), device=device)
    img = torch.nn.Parameter(initial_tensor)
    
    model = model.to(device)
    for it in tqdm(range(n_it), desc='Gradient ascent in image space'):

        out = compute_activations_for_gradient_ascent(
            model, img, layer=layer, filter_id=filter_id
            )

        loss = - out.mean() + wd * torch.norm(img)  # TODO: complete.

        # compute gradient
        loss.backward()  # TODO: complete.

        # normalize gradient
        grads = img.grad.data  # TODO: complete.
        grads = grads.div(torch.norm(grads)+1e-8)

        # Update image
        img.data = img.data + lr * grads  # TODO: complete.
        img.grad.zero_()  # TODO: complete.

        # Apply gaussian blur
        if it % reg_step == 0:
            img_np = torch.squeeze(img).detach().cpu().numpy().transpose((2, 1, 0))
            blurred_img = gaussian_filter(img_np, sigma=(0.3, 0.3, 0))
            
            blurred_tensor = torch.from_numpy(blurred_img).float().transpose(2, 0)
            blurred_tensor = torch.unsqueeze(blurred_tensor, 0).to(device)
            
            img.data = blurred_tensor

            # img = gaussian_filter(torch.squeeze(img).detach().cpu().numpy().transpose((2, 1, 0)),
            #                         sigma=(0.3, 0.3, 0))
            # img = torch.unsqueeze(torch.from_numpy(img).float().transpose(2, 0), 0)
            # img = torch.nn.Parameter(data=img).to(device)

    return deprocess_image(img.detach().cpu().numpy())


##################################
# --- Ajout du top-k filters --- #
##################################
import numpy as np

def compute_filter_activations(model, dataset, filter_ids, layer=5, batch_size=32, device='cuda'):
    """
    Calcule la moyenne spatiale de chaque filtre pour toutes les images du dataset.
    
    Returns:
        activations_dict : dict {filter_id: list of mean activations per image}
        indices : list of dataset indices corresponding to activations
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    activations_dict = {f: [] for f in filter_ids}
    indices = []

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        batch_activation = compute_batch_activations(model, data, layer)  # [B,C,H,W]

        for f in filter_ids:
            # moyenne spatiale par image
            acts = batch_activation[:, f, :, :].mean(dim=(1,2))
            activations_dict[f].extend(acts.detach().cpu().numpy())

        indices.extend(np.arange(batch_idx*batch_size, batch_idx*batch_size + data.size(0)))

    return activations_dict, indices

def get_topk_images(activations_dict, indices, k=10):
    """
    Retourne les indices des Top-k images pour chaque filtre.
    """
    topk_indices = {}
    for f, acts in activations_dict.items():
        acts = np.array(acts)
        sorted_idx = np.argsort(-acts)  # tri décroissant
        topk_indices[f] = [indices[i] for i in sorted_idx[:k]]
    return topk_indices


