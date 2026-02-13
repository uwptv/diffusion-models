import mlflow.pytorch
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get tinyhar for feature extraction
TinyHAR = mlflow.pytorch.load_model("models:/TinyHAR/2").to(device)


def extract_features(data: torch.Tensor, batch_size: int = 250) -> torch.Tensor:
    """
    Extract features from data using the TinyHAR encoder.
    Args:
    - data: tuple of (data_tensor, labels_tensor) where data_tensor is of shape (N, C, L) and labels_tensor is of shape (N,)
    - batch_size: Batch size for feature extraction
    Returns:
    - features: tensor of shape (B, feature_dim)
    """
    features = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size].to(device)
            feat = TinyHAR.encode(batch)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)
