import torch

from .density_coverage import compute_density_coverage
from .feature_encoding import extract_features
from .fid import compute_fid
from .improved_pr import compute_improved_pr
from .kid import compute_kid


def compute_all_metrics(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    use_toy: bool,
    batch_size: int = 250,
) -> dict[str, float]:
    """
    Compute all implemented quality metrics for generated samples.

    Args:
        real_data: Real data samples (N, C, L)
        generated_data: Generated data samples (M, C, L)
        use_toy: Whether to use the toy TinyHAR model
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with all metrics, ready for mlflow.log_metrics()
        Keys: fid, kid_mean, kid_std, precision, recall, density, coverage
    """
    metrics = {}

    # Compute embeddings
    with torch.no_grad():
        real_features = extract_features(
            real_data, batch_size=batch_size, use_toy=use_toy
        )
        gen_features = extract_features(
            generated_data, batch_size=batch_size, use_toy=use_toy
        )

    # FID
    fid_dict = compute_fid(
        real_features=real_features,
        gen_features=gen_features,
    )
    metrics["fid"] = fid_dict["fid"]

    # KID
    kid_mean, kid_std = compute_kid(
        real_features=real_features,
        gen_features=gen_features,
    )
    metrics["kid_mean"] = kid_mean
    metrics["kid_std"] = kid_std

    # Precision & Recall
    precision, recall = compute_improved_pr(
        real_features=real_features,
        gen_features=gen_features,
    )
    metrics["precision"] = precision
    metrics["recall"] = recall

    # Density & Coverage
    density, coverage = compute_density_coverage(
        real_features=real_features,
        gen_features=gen_features,
    )
    metrics["density"] = density
    metrics["coverage"] = coverage

    return metrics
