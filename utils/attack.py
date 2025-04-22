import torch
from utils.utils import atleast_kdim
import torch.nn.functional as F

def distance(a, b):
    return (a - b).flatten(1).norm(dim=1)


def get_init_with_noise(model, X, y):
    init = X.clone()
    p = model(X).argmax(1)

    while any(p == y):
        init = torch.where(
            atleast_kdim(p == y, len(X.shape)), 
            (X + 0.5*torch.randn_like(X)).clip(0, 1), # Add Gaussian noise to the sample
            init)
        p = model(init).argmax(1)
    return init

def get_multiple_orthogonal_directions(X, n_directions=10):
    """
    Generate n orthogonal directions per input in the batch.
    Returns a list of direction tensors of shape (batch_size, C, H, W).
    """
    batch_size = X.size(0)
    flat_X = X.view(batch_size, -1)
    directions = []

    for _ in range(n_directions):
        rand = torch.randn_like(flat_X)
        proj = (rand * flat_X).sum(1, keepdim=True) / (flat_X.norm(dim=1, keepdim=True)**2 + 1e-8)
        orth = rand - proj * flat_X
        orth = F.normalize(orth, p=2, dim=1)
        directions.append(orth.view_as(X))

    return directions  # List of tensors (each direction)

def get_init_with_orthogonal_noise(model, X, y, n_directions=10, epsilon=0.1):
    """
    For each sample in X, generate n_directions orthogonal to it,
    apply perturbations, and select the best adversarial candidate
    (i.e., one that fools the model with minimal L2 distance).
    """
    batch_size = X.size(0)
    best_init = get_init_with_noise(model,X,y)
    best_dist = torch.full((batch_size,), float('inf'), device=X.device)

    directions = get_multiple_orthogonal_directions(X, n_directions)

    for dir in directions:
        candidate = (X + epsilon * dir).clamp(0, 1)
        preds = model(candidate).argmax(1)
        is_adv = preds != y
        dist = (candidate - X).view(batch_size, -1).norm(p=2, dim=1)

        # Update best candidate where this one is better
        better = is_adv & (dist < best_dist)
        better_expanded = atleast_kdim(better, X.dim())

        best_init = torch.where(better_expanded, candidate, best_init)
        best_dist = torch.where(better, dist, best_dist)

    return best_init