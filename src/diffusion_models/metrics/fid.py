import torch


def _compute_stats(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = features.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples, got {n}")

    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    cov = (centered.T @ centered) / (n - 1)
    return mean, cov


def _sqrtm_sym(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=eps)
    sqrt_eigvals = torch.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def compute_fid(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    mu_r, cov_r = _compute_stats(real_features)
    mu_g, cov_g = _compute_stats(gen_features)

    diff = mu_r - mu_g
    mean_term = diff.dot(diff)

    cov_r = cov_r + torch.eye(cov_r.shape[0], device=cov_r.device) * eps
    cov_g = cov_g + torch.eye(cov_g.shape[0], device=cov_g.device) * eps

    sqrt_cov_r = _sqrtm_sym(cov_r, eps=eps)
    cov_prod = sqrt_cov_r @ cov_g @ sqrt_cov_r
    sqrt_cov_prod = _sqrtm_sym(cov_prod, eps=eps)

    fid = (
        mean_term
        + torch.trace(cov_r)
        + torch.trace(cov_g)
        - 2.0 * torch.trace(sqrt_cov_prod)
    )
    return float(fid.item())
