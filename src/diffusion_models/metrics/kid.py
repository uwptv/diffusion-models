import torch


def _polynomial_mmd_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> torch.Tensor:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef0) ** degree


def compute_kid(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    subset_size: int = 1000,
    num_subsets: int = 10,
    degree: int = 3,
) -> tuple[float, float]:
    n_real = int(real_features.shape[0])
    n_gen = int(gen_features.shape[0])

    if subset_size < 2:
        raise ValueError(f"subset_size must be >= 2, got {subset_size}")
    if num_subsets < 1:
        raise ValueError(f"num_subsets must be >= 1, got {num_subsets}")

    # Safe effective size: can always be sampled without replacement from both sets.
    m = min(subset_size, n_real, n_gen)
    if m < 2:
        raise ValueError(
            f"KID needs at least 2 samples in each set, got n_real={n_real}, n_gen={n_gen}"
        )

    # Use float64 for numerical stability in kernel/MMD computations.
    real = real_features.double()
    gen = gen_features.double()

    kid_scores = []
    for _ in range(num_subsets):
        real_idx = torch.randperm(n_real, device=real.device)[:m]
        gen_idx = torch.randperm(n_gen, device=gen.device)[:m]

        real_subset = real[real_idx]
        gen_subset = gen[gen_idx]

        k_rr = _polynomial_mmd_kernel(real_subset, real_subset, degree=degree)
        k_gg = _polynomial_mmd_kernel(gen_subset, gen_subset, degree=degree)
        k_rg = _polynomial_mmd_kernel(real_subset, gen_subset, degree=degree)

        # Unbiased MMD^2 estimator with actual sampled size m.
        kid = (k_rr.sum() - k_rr.trace()) / (m * (m - 1))
        kid += (k_gg.sum() - k_gg.trace()) / (m * (m - 1))
        kid -= 2.0 * k_rg.mean()

        kid_scores.append(kid)

    kid_scores = torch.stack(kid_scores)
    kid_mean = float(kid_scores.mean().item())

    # Avoid NaN std when num_subsets == 1.
    std_correction = 1 if kid_scores.numel() > 1 else 0
    kid_std = float(kid_scores.std(correction=std_correction).item())

    return kid_mean, kid_std
