import torch

from .density_coverage import compute_density_coverage
from .feature_encoding import extract_features
from .fid import compute_fid
from .improved_pr import compute_improved_pr
from .kid import compute_kid


def compute_all_metrics(
    real_data: list[torch.Tensor],
    generated_data: list[torch.Tensor],
    use_toy: bool,
    batch_size: int = 250,
) -> dict[str, float]:
    """
    Compute all implemented quality metrics for generated samples per class and averaged.

    Args:
        real_data: List of real data samples per class, each (N_i, C, L)
        generated_data: List of generated data samples per class, each (M_i, C, L)
        use_toy: Whether to use the toy TinyHAR model
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with per-class and average metrics.
        Keys: fid_class_1, fid_class_2, ..., fid_avg, kid_mean_avg, etc.
    """
    num_classes = len(real_data)
    metrics = {}
    per_class_metrics = {i: {} for i in range(num_classes)}

    # Compute metrics per class
    for class_idx in range(num_classes):
        with torch.no_grad():
            real_features = extract_features(
                real_data[class_idx], batch_size=batch_size, use_toy=use_toy
            )
            gen_features = extract_features(
                generated_data[class_idx], batch_size=batch_size, use_toy=use_toy
            )

        # FID
        fid = compute_fid(real_features=real_features, gen_features=gen_features)
        per_class_metrics[class_idx]["fid"] = fid
        metrics[f"fid_class_{class_idx + 1}"] = fid  # class_idx + 1 for class numbering

        # KID
        kid_mean, kid_std = compute_kid(
            real_features=real_features, gen_features=gen_features
        )
        per_class_metrics[class_idx]["kid_mean"] = kid_mean
        per_class_metrics[class_idx]["kid_std"] = kid_std
        metrics[f"kid_mean_class_{class_idx + 1}"] = kid_mean
        metrics[f"kid_std_class_{class_idx + 1}"] = kid_std

        # Precision & Recall
        precision, recall = compute_improved_pr(
            real_features=real_features, gen_features=gen_features
        )
        per_class_metrics[class_idx]["precision"] = precision
        per_class_metrics[class_idx]["recall"] = recall
        metrics[f"precision_class_{class_idx + 1}"] = precision
        metrics[f"recall_class_{class_idx + 1}"] = recall

        # Density & Coverage
        density, coverage = compute_density_coverage(
            real_features=real_features, gen_features=gen_features
        )
        per_class_metrics[class_idx]["density"] = density
        per_class_metrics[class_idx]["coverage"] = coverage
        metrics[f"density_class_{class_idx + 1}"] = density
        metrics[f"coverage_class_{class_idx + 1}"] = coverage

    # Compute averages across classes
    metric_names = [
        "fid",
        "kid_mean",
        "kid_std",
        "precision",
        "recall",
        "density",
        "coverage",
    ]
    for metric_name in metric_names:
        values = [per_class_metrics[i][metric_name] for i in range(num_classes)]
        avg = sum(values) / len(values)
        metrics[f"{metric_name}_avg"] = avg

    return metrics
