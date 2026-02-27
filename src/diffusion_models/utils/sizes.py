import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

MiB = 1024**2
GigaFLOP = 10**9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


def model_flops(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device | None = None,
) -> int:
    """
    Compute FLOPs (floating point operations) for a model forward pass
    using torch.profiler instead of fvcore to avoid JIT tracing issues.

    Args:
        model: PyTorch model
        input_shape: Tuple of (shape, dtype) pairs or just shapes.
                     e.g., ((1, 3, 128), (1, 1, 1), (1,))
        device: Device to run the computation on

    Returns:
        Total FLOPs as integer
    """
    if device is None:
        device = next(model.parameters()).device

    # Create dummy inputs matching actual training dtypes
    dummy_inputs = []
    for idx, shape in enumerate(input_shape):
        if idx == len(input_shape) - 1:
            # Last input is class label - Long tensor
            dummy_inputs.append(torch.zeros(*shape, dtype=torch.long, device=device))
        else:
            dummy_inputs.append(torch.randn(*shape, device=device))

    model.eval()

    with (
        torch.no_grad(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True,
        ) as prof,
    ):
        model(*dummy_inputs)

    total_flops = sum(
        event.flops for event in prof.key_averages() if event.flops is not None
    )

    return total_flops


def count_flops(model, channels: int, seq_len: int, batch_size=1, device=device):
    """
    Calculates the FLOPs for a U-Net with (data, time, class) inputs.
    """
    # 1. Put model in eval mode (critical for BatchNorm/Dropout behavior)
    model.eval()
    model.to("cpu")

    # 2. Construct dummy inputs matching your architecture
    # x: (batch, channels, length)
    x_dummy = torch.randn(batch_size, channels, seq_len)

    # t: (batch, 1, 1) - Diffusion timestep
    t_dummy = torch.randn(batch_size, 1, 1)

    # y: (batch, 1) - Class labels (int64/long)
    y_dummy = torch.randint(0, 4, (batch_size, 1))

    # 3. Use fvcore to analyze the forward pass
    # Inputs must be a tuple in the exact order of your forward(self, x, t, y)
    inputs = (x_dummy, t_dummy, y_dummy)

    try:
        flops_analyzer = FlopCountAnalysis(model, inputs)
        total_flops = flops_analyzer.total()

        return total_flops

    except Exception as e:
        print(f"FLOP calculation failed: {e}")
        return 0.0
