from dataclasses import dataclass, field

from torch_modules import Device


@dataclass(frozen=True)
class TorchCodecWriteOptions:
    """
    Settings specific to `TorchCodecVideoWriter`, passed through to FFmpeg.

    Attributes
    ----------
    device : Device
        Device to encode on; CUDA encodes on the GPU's NVENC hardware. A device
        `torchcodec` cannot use here falls back to ``Device.CPU`` with a warning.
    codec : str | None
        FFmpeg encoder name such as ``"libx264"``; None uses the container's
        default.
    pixel_format : str | None
        Encoded pixel format such as ``"yuv420p"``; None uses the codec's
        default. Must be None when encoding on CUDA.
    crf : float | None
        Constant Rate Factor, lower is better quality; None uses the encoder's
        default.
    preset : str | None
        Speed/compression trade-off such as ``"fast"``; None uses the
        encoder's default.
    batch_size : int
        Frames buffered before each hand-off to the encoder.
    """

    device: Device = field(default_factory=Device.detect)
    codec: str | None = None
    pixel_format: str | None = None
    crf: float | None = None
    preset: str | None = None
    batch_size: int = 16

    def __post_init__(self) -> None:
        """
        Validate the options.

        Raises
        ------
        ValueError
            If `batch_size` is not positive.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
