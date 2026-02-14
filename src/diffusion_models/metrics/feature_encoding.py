import mlflow.pytorch
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get tinyhar for feature extraction
TinyHAR = mlflow.pytorch.load_model("models:/TinyHAR/2").to(device)


def extract_features(data: torch.Tensor, batch_size: int = 250) -> torch.Tensor:
    """
    Extract features from data using the TinyHAR encoder.
    Args:
    - data: tensor of shape (N, C, L) containing raw sensor data
    - batch_size: Batch size for feature extraction
    Returns:
    - features: tensor of shape (N, feature_dim)
    """
    features = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size].to(device)  # (batch_size, C, L)
            feat = TinyHAR.encode(batch)  # (batch_size, feature_dim)
            features.append(feat.cpu())  # [(batch_size, feature_dim), ...]
    return torch.cat(features, dim=0)
