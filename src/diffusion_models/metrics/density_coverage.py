import torch


def compute_density_coverage(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    k: int = 5,
    batch_size: int = 1000,
) -> tuple[float, float]:
    """
    Compute Density and Coverage metrics.

    For each generated sample, find its k nearest real neighbors.
    - Density: average number of real k-NN spheres that contain each generated sample (normalized by k)
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

    # For density & coverage: compute k-NN radii of real samples
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

    # Density: average count of real k-NN spheres that contain each generated sample, normalized by k
    gen_density_counts = []
    for i in range(0, n_gen, batch_size):
        gen_chunk = gen_features[i : i + batch_size]
        dist = torch.cdist(gen_chunk, real_features)  # (B, N_real)
        counts = (dist <= real_radii.unsqueeze(0)).sum(dim=1)  # (B,)
        gen_density_counts.append(counts)

    gen_density_counts = torch.cat(gen_density_counts, dim=0)
    density = (gen_density_counts.float().mean() / k).item()

    # Coverage: fraction of real samples with a generated neighbor
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
