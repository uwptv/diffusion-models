import torch


def compute_pairwise_distance(data_x: torch.Tensor, data_y: torch.Tensor | None = None):
    """
    Args:
        data_x: torch.Tensor([N, feature_dim], dtype=torch.float32)
        data_y: torch.Tensor([M, feature_dim], dtype=torch.float32)
    Returns:
        torch.Tensor([N, M], dtype=torch.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x

    # Compute pairwise Euclidean distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y^T
    x_norm = (data_x**2).sum(dim=1, keepdim=True)  # [N, 1]
    y_norm = (data_y**2).sum(dim=1, keepdim=True)  # [M, 1]

    dists = x_norm + y_norm.T - 2.0 * torch.mm(data_x, data_y.T)

    # Clamp to avoid numerical errors with sqrt of negative numbers
    dists = torch.clamp(dists, min=0.0)
    dists = torch.sqrt(dists)

    return dists


def get_kth_value(unsorted: torch.Tensor, k: int, dim=-1):
    """
    Args:
        unsorted: torch.Tensor of any dimensionality.
        k: int
    Returns:
        kth values along the designated dimension.
    """
    # Get the k smallest values along the specified dimension
    # kthvalue returns (values, indices), we only need values
    k_smallests, _ = torch.topk(unsorted, k, dim=dim, largest=False, sorted=False)

    # Get the maximum of these k smallest values (which is the k-th smallest)
    kth_values = k_smallests.max(dim=dim)[0]

    return kth_values


def compute_nearest_neighbour_distances(input_features: torch.Tensor, nearest_k: int):
    """
    Args:
        input_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        nearest_k: int
    Returns:
        torch.Tensor([N], dtype=torch.float32) of distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, dim=-1)
    return radii


def compute_dc(real_features, fake_features, nearest_k):
    """
    Computes density and coverage given two manifolds.

    Args:
        real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        fake_features: torch.Tensor([M, feature_dim], dtype=torch.float32)
        nearest_k: int.
    Returns:
        dict of density and coverage.
    """

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k
    )  # (N,)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features
    )  # shape (N, M)

    density = (1.0 / float(nearest_k)) * (
        (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1))
        .sum(dim=0)
        .float()
        .mean()
        .item()
    )

    coverage = (
        (distance_real_fake.min(dim=1)[0] < real_nearest_neighbour_distances)
        .float()
        .mean()
        .item()
    )

    return dict(density=density, coverage=coverage)
