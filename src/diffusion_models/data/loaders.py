from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from whar_datasets import (
    KFoldSplitter,
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
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
        balanced: bool = True,
    ):
        super().__init__()
        # create cfg for dataset
        dataset_id = WHARDatasetID(dataset)
        cfg = get_dataset_cfg(dataset_id)
        # cfg.parallelize = True
        if dataset_id == WHARDatasetID.UCI_HAR:
            cfg.sensor_channels = ["body_acc_x", "body_acc_y", "body_acc_z"]
            cfg.window_time = 2.56  # 128 timesteps at 50Hz
        else:
            cfg.window_time = window_time
        cfg.seed = seed

        # create and run pre-processing pipeline
        pre_pipeline = PreProcessingPipeline(cfg)
        activity_df, session_df, window_df = pre_pipeline.run()

        # create KFOLD splits
        splitter = KFoldSplitter(cfg)
        splits = splitter.get_splits(session_df, window_df)
        self.split = splits[0]

        # create and run post-processing pipeline for the specific split
        post_pipeline = PostProcessingPipeline(
            cfg, pre_pipeline, window_df, self.split.train_indices
        )
        samples = post_pipeline.run()

        # create dataloaders for the specific split
        self.loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
        self.dummy = nn.Buffer(
            torch.zeros(1)
        )  # Will automatically be moved when self.to(...) is called
        self.seed = seed
        self.balanced = balanced
        self.num_classes = cfg.num_of_activities
        self._gen = torch.Generator()  # CPU generator
        if self.seed is not None:
            self._gen.manual_seed(self.seed)

    def reset_generator(self) -> None:
        self._gen = torch.Generator()
        if self.seed is not None:
            self._gen.manual_seed(self.seed)

    def sample(
        self,
        num_samples: int,
        subset: str = "train",
        class_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
            - subset: which set to sample from ("train", "val", or "test")
            - class_idx: if provided, only sample from this class (0-indexed). If None, sample from all classes.
        Returns:
            - samples: shape (batch_size, channels, signal_length)
            - labels: shape (batch_size, label_dim = 1) with values corresponding to activity class indices
        """
        if subset == "train":
            indices = self.split.train_indices
        elif subset == "val":
            indices = self.split.val_indices
        elif subset == "test":
            indices = self.split.test_indices
        else:
            raise ValueError(f"Invalid subset: {subset}")

        samples_list = []
        labels_list = []
        num_classes = self.num_classes

        if class_idx is not None and (class_idx < 0 or class_idx > num_classes - 1):
            raise ValueError(
                f"Invalid class_idx: {class_idx}. Must be between 0 and {num_classes - 1}."
            )
        elif class_idx is not None:
            activity_labels, _, samples = self.loader.sample_items(
                num_samples, indices=indices, activity_id=class_idx
            )
            samples_list.append(samples)
            labels_list.append(activity_labels)
        else:
            # Default behavior: sample as evenly as possible from all classes.
            # Any remainder is distributed to the first classes.
            base = num_samples // num_classes
            remainder = num_samples % num_classes

            for activity_idx in range(num_classes):
                n_for_class = base + (1 if activity_idx < remainder else 0)
                if n_for_class == 0:
                    continue

                activity_labels, _, samples = self.loader.sample_items(
                    n_for_class,
                    indices=indices,
                    activity_id=activity_idx,
                    seed=self.seed,
                )
                samples_list.append(samples)
                labels_list.append(activity_labels)
        if not samples_list:
            raise ValueError(
                "No samples were collected. Check num_samples and filters."
            )

        samples_np = np.concatenate(
            [np.asarray(sample_batch) for sample_batch in samples_list], axis=0
        )
        labels_np = np.concatenate(
            [np.asarray(label_batch) for label_batch in labels_list], axis=0
        )

        # Convert to tensors of appropriate shape
        # Use from_numpy to avoid an extra CPU copy before moving to target device.
        samples = torch.from_numpy(samples_np).to(
            device=self.dummy.device, dtype=torch.float32
        )  # (batch_size, 1, signal_length, channels)
        samples = samples.squeeze(1).permute(
            0, 2, 1
        )  # (batch_size, signal_length, channels)

        activity_labels = (
            torch.from_numpy(labels_np)
            .to(device=self.dummy.device, dtype=torch.int64)
            .unsqueeze(1)
        )  # shape (batch_size, 1)

        # Shuffle samples and labels with the same permutation.
        permutation = torch.randperm(
            samples.size(0),
            generator=self._gen,
        ).to(self.dummy.device)
        samples = samples[permutation]
        activity_labels = activity_labels[permutation]

        return samples, activity_labels

    def showcase_class_imbalance(self, plot: bool = True) -> dict[str, object]:
        """
        Summarize overall class distribution across all available split indices
        (train + val + test combined).

        Returns:
            {
                "counts": {class_id: n, ...},
                "percentages": {class_id: p, ...},
                "imbalance_ratio": r,
                "total": total_samples,
            }
        """
        class_ids = list(range(self.num_classes))

        # Combine indices from all splits once (deduplicated)
        all_indices = sorted(
            set(self.split.train_indices)
            | set(self.split.val_indices)
            | set(self.split.test_indices)
        )

        counts: dict[int, int] = {}
        total = 0

        for class_id in class_ids:
            class_indices = self.loader.filter_indices(
                indices=all_indices, activity_id=class_id
            )
            n = len(class_indices)
            counts[class_id] = n
            total += n

        percentages = {
            c: (100.0 * n / total if total > 0 else 0.0) for c, n in counts.items()
        }

        non_zero = [n for n in counts.values() if n > 0]
        if len(non_zero) >= 2:
            imbalance_ratio = max(non_zero) / min(non_zero)
        else:
            imbalance_ratio = float("inf") if len(non_zero) == 1 else 0.0

        result = {
            "counts": counts,
            "percentages": percentages,
            "imbalance_ratio": imbalance_ratio,
            "total": total,
        }

        print(f"\n[all] total={total}, imbalance_ratio={imbalance_ratio:.3f}")
        for c in class_ids:
            print(f"  class {c}: {counts[c]} ({percentages[c]:.2f}%)")

        if plot:
            x = np.arange(self.num_classes)
            all_counts = [counts[c] for c in class_ids]
            all_percentages = [percentages[c] for c in class_ids]
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

            plt.figure(figsize=(10, 5))
            bars = plt.bar(x, all_counts, width=0.6, color=colors)
            plt.xticks(x, [str(c) for c in class_ids])
            plt.xlabel("Class")
            plt.ylabel("Number of samples")
            plt.title("Overall class distribution with class percentages")

            # Add percentage labels above bars.
            for bar, pct in zip(bars, all_percentages):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()

        return result


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


if __name__ == "__main__":
    sampler = DataSampler()
    samples, labels = sampler.sample(num_samples=4, subset="train")
    visualize_wisdm_samples(samples, labels, num_plots=4)
