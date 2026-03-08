import torch
import torch.nn.functional as F
import pdb


def nce_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 1.0
        ) -> torch.Tensor:
    """
    PyTorch implementation of the NT-Xent loss introduced in 
    https://proceedings.mlr.press/v119/chen20j/chen20j.pdf

    Args:
        z1: embedding from view 1 (Tensor) of shape (bsz, dim).
        z2: embedding from view 2 (Tensor) of shape (bsz, dim).
        temperature: a floating number for temperature scaling.
    """
    LARGE_NUM = 1e9
    SMALL_NUM = 1e-9

    # TODO: implement NT-Xent loss

    # concaténer les embeddings de deux vues
    z = torch.cat([z1, z2], dim=0)

    # Normalisation L2
    z = F.normalize(z, dim=1)

    # Similarité cosinus entre tous les éléments du batch
    sim = torch.mm(z, z.t()) / temperature

    # Masquer les similarités entre les éléments du même batch
    # exemple diagonal de sim correspond à la similarité entre les éléments du même batch, on les masque en mettant une grande valeur négative
    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim = sim.masked_fill(mask, -LARGE_NUM)

    # Cibles : les éléments du même batch sont des paires positives
    n = z1.size(0)  # taille du batch
    targets = torch.arange(n, device=sim.device)
    targets = torch.cat([targets + n, targets], dim=0)

    # Calcul de la loss cross-entropy
    loss = F.cross_entropy(sim, targets)

    return loss