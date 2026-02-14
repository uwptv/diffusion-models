import torch

from .feature_encoding import extract_features


def _compute_density_coverage(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    k: int = 5,
    batch_size: int = 1000,
) -> tuple[float, float]:
    """
    Compute Density and Coverage metrics.

    For each generated sample, find its k nearest real neighbors.
    - Density: average distance to k-th nearest real neighbor (lower is better)
    - Coverage: fraction of real samples that have a generated neighbor within their k-NN radius

    Args:
        real_features: Real feature vectors (N, D)
        gen_features: Generated feature vectors (M, D)
        k: kNN neighborhood size
        batch_size: Batch size for chunked distance computation

    Returns:
        (density, coverage) where density is in [0, inf) and coverage is in [0, 1]
    """
    n_real = real_features.shape[0]
    n_gen = gen_features.shape[0]

    if k >= n_real:
        raise ValueError(f"k={k} must be < n_real={n_real}")

    # Compute k-NN distances from generated to real
    gen_to_real_dists = []
    for i in range(0, n_gen, batch_size):
        gen_chunk = gen_features[i : i + batch_size]
        dist = torch.cdist(gen_chunk, real_features)
        # Get k smallest distances (k-th is at index k-1)
        knn_dists = torch.topk(dist, k, largest=False).values[:, -1]
        gen_to_real_dists.append(knn_dists)

    gen_to_real_dists = torch.cat(gen_to_real_dists, dim=0)
    density = gen_to_real_dists.mean().item()

    # For coverage: compute k-NN radii of real samples
    real_radii = []
    for i in range(0, n_real, batch_size):
        real_chunk = real_features[i : i + batch_size]
        dist = torch.cdist(real_chunk, real_features)

        # Exclude self-distance
        idx = torch.arange(i, i + real_chunk.shape[0], device=real_features.device)
        dist[torch.arange(real_chunk.shape[0], device=real_features.device), idx] = (
            float("inf")
        )

        knn_dists = torch.topk(dist, k, largest=False).values[:, -1]
        real_radii.append(knn_dists)

    real_radii = torch.cat(real_radii, dim=0)

    # Check coverage: fraction of real samples with a generated neighbor
    covered = 0
    for i in range(0, n_real, batch_size):
        real_chunk = real_features[i : i + batch_size]
        dist = torch.cdist(real_chunk, gen_features)
        is_covered = (dist <= real_radii[i : i + real_chunk.shape[0]].unsqueeze(1)).any(
            dim=1
        )
        covered += is_covered.sum().item()

    coverage = covered / n_real

    return density, coverage


def compute_density_coverage(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    batch_size: int = 250,
    k: int = 5,
    metric_batch_size: int = 1000,
) -> dict[str, float]:
    """
    Compute Density & Coverage with automatic feature extraction.

    Args:
        real_data: Raw real data (N, C, L)
        generated_data: Raw generated data (M, C, L)
        batch_size: Batch size for encoder
        k: kNN neighborhood size
        metric_batch_size: Batch size for distance computations

    Returns:
        {"density": float, "coverage": float}
    """
    real_features = extract_features(real_data, batch_size=batch_size)
    gen_features = extract_features(generated_data, batch_size=batch_size)

    density, coverage = _compute_density_coverage(
        real_features=real_features,
        gen_features=gen_features,
        k=k,
        batch_size=metric_batch_size,
    )

    return {"density": density, "coverage": coverage}
