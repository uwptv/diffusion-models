import torch


def _kth_neighbor_distance(
    features: torch.Tensor, k: int, batch_size: int
) -> torch.Tensor:
    """
    Compute the distance to the k-th nearest neighbor for each feature vector.
    Args:
    - features: tensor of shape (N, D) containing feature vectors
    - k: neighborhood size for kNN
    - batch_size: Batch size for chunked distance computation
    Returns:
    - kth: tensor of shape (N,) containing distance to k-th nearest neighbor
    """
    n = features.shape[0]
    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")

    kth = torch.empty(n, device=features.device)

    for i in range(0, n, batch_size):
        chunk = features[i : i + batch_size]  # (batch_size, feature_dim)
        dist = torch.cdist(chunk, features)  # (batch_size, n)

        # Exclude self-distance for the same-set kNN query
        idx = torch.arange(i, i + chunk.shape[0], device=features.device)
        dist[torch.arange(chunk.shape[0], device=features.device), idx] = float("inf")

        kth[i : i + chunk.shape[0]] = torch.topk(dist, k, largest=False).values[
            :, -1
        ]  # (batch_size,)

    return kth


def _coverage(
    queries: torch.Tensor,
    refs: torch.Tensor,
    ref_radii: torch.Tensor,
    batch_size: int,
) -> float:
    n = queries.shape[0]
    covered = 0

    for i in range(0, n, batch_size):
        q_chunk = queries[i : i + batch_size]
        dist = torch.cdist(q_chunk, refs)

        # A query is covered if it lies within any reference radius
        is_covered = (dist <= ref_radii.unsqueeze(0)).any(dim=1)
        covered += is_covered.sum().item()

    return covered / n


def compute_improved_pr(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    k: int = 3,
    batch_size: int = 1000,
) -> dict[str, float]:
    """
    Improved precision/recall for generative models (Kynkaanniemi et al., 2019).

    Args:
        real_features: Real feature vectors (N, D)
        gen_features: Generated feature vectors (M, D)
        k: kNN neighborhood size
        batch_size: Batch size for chunked distance computation

    Returns:
        precision, recall as floats in [0, 1]
    """
    real_radii = _kth_neighbor_distance(real_features, k, batch_size)
    gen_radii = _kth_neighbor_distance(gen_features, k, batch_size)

    precision = _coverage(gen_features, real_features, real_radii, batch_size)
    recall = _coverage(real_features, gen_features, gen_radii, batch_size)

    return precision, recall
