import torch

from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath

from .density_coverage import compute_dc
from .feature_encoding import extract_features
from .fid import compute_fid
from .improved_pr import compute_pr
from .kid import compute_kid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_all_metrics(
    model: torch.nn.Module,
    path: GaussianConditionalProbabilityPath,
    num_classes: int,
    guidance_scales: list[float],
    evaluator: str,
    batch_size: int = 250,
) -> dict[str, float]:
    """
    Compute all implemented quality metrics for generated samples per guidance scale and per class.

    Args:
        model: The trained model to evaluate
        guidance_scales: List of guidance scales to evaluate
        evaluator: The evaluator to use for feature extraction (e.g., "wisdm", "toy", "resnet18").
            This determines the feature space in which metrics are computed.
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with metrics organized by guidance scale and class (0-indexed).
        Keys: fid_gs2.0_class_0, fid_gs2.0_class_1, ..., fid_gs2.0_avg,
              fid_gs3.0_class_0, ..., fid_gs3.0_avg
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        guidance_real_data = []
        guidance_generated_data = []

        # Sample real data once for all guidance scales
        real_data_all_classes = []
        for class_idx in range(num_classes):
            real_sensor_data, _ = path.p_data.sample(
                10000, class_idx=class_idx, subset="test"
            )
            real_data_all_classes.append(real_sensor_data)

        guidance_real_data.append(real_data_all_classes)

        # Generate samples for each guidance scale
        for guidance_scale in guidance_scales:
            generated_per_scale = [
                model.sample(
                    10000,
                    p_data_shape=path.p_simple_shape,
                    class_idx=class_idx,
                    guidance_scale=guidance_scale,
                )
                for class_idx in range(num_classes)
            ]
            guidance_generated_data.append(generated_per_scale)

    metrics = {}
    real_data = guidance_real_data[0]
    generated_data = guidance_generated_data

    for scale_idx, guidance_scale in enumerate(guidance_scales):
        per_class_metrics = {i: {} for i in range(num_classes)}

        for class_idx in range(num_classes):
            with torch.no_grad():
                real_features = extract_features(
                    real_data[class_idx], batch_size=batch_size, evaluator=evaluator
                )
                gen_features = extract_features(
                    generated_data[scale_idx][class_idx],
                    batch_size=batch_size,
                    evaluator=evaluator,
                )

            fid = compute_fid(real_features=real_features, gen_features=gen_features)
            per_class_metrics[class_idx]["fid"] = fid
            metrics[f"fid_gs{guidance_scale}_class_{class_idx}"] = fid

            kid_mean, kid_std = compute_kid(
                real_features=real_features, gen_features=gen_features
            )
            per_class_metrics[class_idx]["kid_mean"] = kid_mean
            per_class_metrics[class_idx]["kid_std"] = kid_std
            metrics[f"kid_mean_gs{guidance_scale}_class_{class_idx}"] = kid_mean
            metrics[f"kid_std_gs{guidance_scale}_class_{class_idx}"] = kid_std

            pr_dict = compute_pr(
                real_features=real_features, fake_features=gen_features, nearest_k=3
            )
            per_class_metrics[class_idx]["precision"] = pr_dict["precision"]
            per_class_metrics[class_idx]["recall"] = pr_dict["recall"]
            metrics[f"precision_gs{guidance_scale}_class_{class_idx}"] = pr_dict[
                "precision"
            ]
            metrics[f"recall_gs{guidance_scale}_class_{class_idx}"] = pr_dict["recall"]

            dc_dict = compute_dc(
                real_features=real_features, fake_features=gen_features, nearest_k=5
            )
            per_class_metrics[class_idx]["density"] = dc_dict["density"]
            per_class_metrics[class_idx]["coverage"] = dc_dict["coverage"]
            metrics[f"density_gs{guidance_scale}_class_{class_idx}"] = dc_dict[
                "density"
            ]
            metrics[f"coverage_gs{guidance_scale}_class_{class_idx}"] = dc_dict[
                "coverage"
            ]

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
