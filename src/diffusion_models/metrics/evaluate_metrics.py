import torch

from .density_coverage import compute_dc
from .feature_encoding import extract_features
from .fid import compute_fid
from .improved_pr import compute_pr
from .kid import compute_kid


def compute_all_metrics(
    real_data: list[torch.Tensor],
    generated_data: list[list[torch.Tensor]],
    used_guidance_scales: list[float],
    use_toy: bool,
    batch_size: int = 250,
) -> dict[str, float]:
    """
    Compute all implemented quality metrics for generated samples per guidance scale and per class.

    Args:
        real_data: List of real data samples per class, each (N_i, C, L)
                   Shape: [num_classes][tensor]
        generated_data: List of generated data per guidance scale per class
                        Shape: [num_guidance_scales][num_classes][tensor]
        used_guidance_scales: List of guidance scales used for generating the data
        use_toy: Whether to use the toy TinyHAR model
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with metrics organized by guidance scale and class.
        Keys: fid_gs2.0_class_1, fid_gs2.0_class_2, ..., fid_gs2.0_avg,
              fid_gs3.0_class_1, ..., fid_gs3.0_avg
    """
    num_classes = len(real_data)
    metrics = {}

    # Compute metrics per guidance scale
    for scale_idx, guidance_scale in enumerate(used_guidance_scales):
        per_class_metrics = {i: {} for i in range(num_classes)}

        # Compute metrics per class for this guidance scale
        for class_idx in range(num_classes):
            with torch.no_grad():
                real_features = extract_features(
                    real_data[class_idx], batch_size=batch_size, use_toy=use_toy
                )
                gen_features = extract_features(
                    generated_data[scale_idx][class_idx],
                    batch_size=batch_size,
                    use_toy=use_toy,
                )

            # FID
            fid = compute_fid(real_features=real_features, gen_features=gen_features)
            per_class_metrics[class_idx]["fid"] = fid
            metrics[f"fid_gs{guidance_scale}_class_{class_idx + 1}"] = fid

            # KID
            kid_mean, kid_std = compute_kid(
                real_features=real_features, gen_features=gen_features
            )
            per_class_metrics[class_idx]["kid_mean"] = kid_mean
            per_class_metrics[class_idx]["kid_std"] = kid_std
            metrics[f"kid_mean_gs{guidance_scale}_class_{class_idx + 1}"] = kid_mean
            metrics[f"kid_std_gs{guidance_scale}_class_{class_idx + 1}"] = kid_std

            # Precision & Recall
            pr_dict = compute_pr(real_features=real_features, gen_features=gen_features)
            per_class_metrics[class_idx]["precision"] = pr_dict["precision"]
            per_class_metrics[class_idx]["recall"] = pr_dict["recall"]
            metrics[f"precision_gs{guidance_scale}_class_{class_idx + 1}"] = pr_dict[
                "precision"
            ]
            metrics[f"recall_gs{guidance_scale}_class_{class_idx + 1}"] = pr_dict[
                "recall"
            ]

            # Density & Coverage
            dc_dict = compute_dc(real_features=real_features, gen_features=gen_features)
            per_class_metrics[class_idx]["density"] = dc_dict["density"]
            per_class_metrics[class_idx]["coverage"] = dc_dict["coverage"]
            metrics[f"density_gs{guidance_scale}_class_{class_idx + 1}"] = dc_dict[
                "density"
            ]
            metrics[f"coverage_gs{guidance_scale}_class_{class_idx + 1}"] = dc_dict[
                "coverage"
            ]

        # Compute averages across classes for this guidance scale
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
            metrics[f"{metric_name}_gs{guidance_scale}_avg"] = avg

    return metrics
