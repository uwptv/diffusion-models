import torch

from .feature_encoding import extract_features


def _polynomial_mmd_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    gamma: float = None,
    coef0: float = 1.0,
) -> torch.Tensor:
    """Polynomial kernel for MMD."""
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    return (gamma * (x @ y.T) + coef0) ** degree


def _compute_kid(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    subset_size: int = 1000,
    num_subsets: int = 10,
    degree: int = 3,
) -> tuple[float, float]:
    """
    Compute Kernel Inception Distance.

    Args:
        real_features: Real feature vectors (N, feature_dim)
        gen_features: Generated feature vectors (M, feature_dim)
        subset_size: Number of samples per subset
        num_subsets: Number of subsets for computing mean/std
        degree: Polynomial kernel degree

    Returns:
        (kid_mean, kid_std)
    """
    n_real = real_features.shape[0]
    n_gen = gen_features.shape[0]

    kid_scores = []
    for _ in range(num_subsets):
        # Random subsets
        real_idx = torch.randperm(n_real)[:subset_size]
        gen_idx = torch.randperm(n_gen)[:subset_size]

        real_subset = real_features[real_idx]
        gen_subset = gen_features[gen_idx]

        # Compute kernels
        k_rr = _polynomial_mmd_kernel(real_subset, real_subset, degree=degree)
        k_gg = _polynomial_mmd_kernel(gen_subset, gen_subset, degree=degree)
        k_rg = _polynomial_mmd_kernel(real_subset, gen_subset, degree=degree)

        # Unbiased KID estimator
        kid = (k_rr.sum() - k_rr.trace()) / (subset_size * (subset_size - 1))
        kid += (k_gg.sum() - k_gg.trace()) / (subset_size * (subset_size - 1))
        kid -= 2 * k_rg.mean()

        kid_scores.append(kid.item())

    kid_scores = torch.tensor(kid_scores)
    return kid_scores.mean().item(), kid_scores.std().item()


def compute_kid(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    batch_size: int = 256,
    **kid_kwargs,
) -> dict[str, float]:
    real_features = extract_features(real_data, batch_size=batch_size)
    gen_features = extract_features(generated_data, batch_size=batch_size)

    kid_mean, kid_std = _compute_kid(real_features, gen_features, **kid_kwargs)

    return {"kid_mean": kid_mean, "kid_std": kid_std}
