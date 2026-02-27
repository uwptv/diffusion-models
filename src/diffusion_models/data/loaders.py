from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from whar_datasets import (
    KFoldSplitter,
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)

from diffusion_models.data.base import Sampleable


class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """

    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root="./datasets",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        self.dummy = nn.Buffer(
            torch.zeros(1)
        )  # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels


class DataSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for a PyTorch DataLoader
    """

    def __init__(
        self,
        dataset: str = "wisdm",
        seed: int = 42,
        window_time: float = 6.0,
        split_type: str = "train",
    ):
        super().__init__()
        # create cfg for dataset
        dataset_id = WHARDatasetID(dataset)
        cfg = get_dataset_cfg(dataset_id)
        # cfg.parallelize = True
        cfg.window_time = window_time
        cfg.seed = seed

        # create and run pre-processing pipeline
        pre_pipeline = PreProcessingPipeline(cfg)
        activity_df, session_df, window_df = pre_pipeline.run()

        # create LOSO splits
        splitter = KFoldSplitter(cfg)
        splits = splitter.get_splits(session_df, window_df)
        split = splits[0]

        # create and run post-processing pipeline for the specific split
        post_pipeline = PostProcessingPipeline(
            cfg, pre_pipeline, window_df, split.train_indices
        )
        samples = post_pipeline.run()

        # create dataloaders for the specific split
        loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
        adapter = TorchAdapter(cfg, loader, split)
        dataloaders = adapter.get_dataloaders(batch_size=64)
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.dummy = nn.Buffer(
            torch.zeros(1)
        )  # Will automatically be moved when self.to(...) is called

    def sample(
        self,
        num_samples: int,
        dataloader: str = "train",
        class_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
            - dataloader: which dataloader to sample from ("train" or "val")
            - class_idx: if provided, only sample from this class (1-6 for WISDM, where 0 is unconditional)
        Returns:
            - samples: shape (batch_size, channels, signal_length)
            - labels: shape (batch_size, label_dim) with values 1-6 (0 reserved for unconditional)
        """
        samples_list = []
        labels_list = []

        if dataloader == "train":
            dl = self.train_dataloader
        elif dataloader == "val":
            dl = self.val_dataloader
        else:
            raise ValueError(f"Unknown dataloader: {dataloader}")

        for batch_labels, batch_samples in dl:
            # Filter by class if class_idx is specified
            if class_idx is not None:
                # batch_labels shape: (batch_size,) or (batch_size, 1)
                mask = batch_labels.squeeze() == (
                    class_idx - 1
                )  # Adjust for 0-based indexing
                batch_samples = batch_samples[mask]
                batch_labels = batch_labels[mask]

            if batch_samples.shape[0] == 0:
                continue  # Skip empty batches

            samples_list.append(batch_samples)
            labels_list.append(batch_labels)

            total_samples = sum(s.shape[0] for s in samples_list)
            if total_samples >= num_samples:
                break

        if len(samples_list) == 0:
            raise ValueError(
                f"No samples found for class {class_idx} in {dataloader} set"
            )

        # Concatenate and slice to exact number
        samples = torch.cat(samples_list, dim=0)[:num_samples].to(self.dummy.device)
        labels = torch.cat(labels_list, dim=0)[:num_samples].to(self.dummy.device)

        # Add 1 to labels to reserve 0 for unconditional generation
        labels = labels + 1

        # Permute samples to have shape (num_samples, channels, signal_length)
        samples = samples.permute(0, 2, 1)

        # Extend labels to have shape (num_samples, label_dim)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        return samples, labels


# samples, labels = DataSampler(dataset="wisdm").sample(10, "val")
# print(f"Samples data: {samples}, Labels data: {labels}")


def visualize_wisdm_samples(
    samples: torch.Tensor, labels: torch.Tensor, num_plots: int = 4
):
    """
    Visualize WISDM time series samples with their activity labels.

    Args:
        samples: shape (batch_size, channels, signal_length)
        labels: shape (batch_size, label_dim) - activity class indices
        num_plots: number of samples to visualize
    """

    # Activity mapping (adjust based on your dataset's activity classes)
    activity_names = {
        0: "Walking",
        1: "Jogging",
        2: "Upstairs",
        3: "Downstairs",
        4: "Sitting",
        5: "Standing",
    }

    num_plots = min(num_plots, samples.shape[0])
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots))

    if num_plots == 1:
        axes = [axes]

    for idx in range(num_plots):
        sample = samples[idx].cpu().detach().numpy()  # shape: (channels, signal_length)
        label = labels[idx].item() if labels[idx].dim() == 0 else labels[idx, 0].item()
        activity_name = activity_names.get(label, f"Activity {label}")

        ax = axes[idx]
        signal_length = sample.shape[1]
        time_axis = range(signal_length)

        # Plot each channel
        channel_names = ["X-axis", "Y-axis", "Z-axis"]
        colors = ["#FF6B6B", "#4ECDC4", "#C8D145"]

        for channel in range(sample.shape[0]):
            ax.plot(
                time_axis,
                sample[channel],
                label=channel_names[channel],
                color=colors[channel],
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Acceleration")
        ax.set_title(f"Activity: {activity_name} (Label: {label})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# visualize_wisdm_samples(samples, labels, num_plots=4)
