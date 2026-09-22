from dataclasses import dataclass, field

from torch_modules import Device


@dataclass(frozen=True)
class TorchCodecReadOptions:
    """
    Settings specific to `TorchCodecVideoReader`.

    Attributes
    ----------
    device : Device
        Device to decode on; CUDA decodes on the GPU's NVDEC hardware. A device
        `torchcodec` cannot use here falls back to ``Device.CPU`` with a warning.
    decode_run_length : int
        Frames decoded per call while iterating. Larger runs amortize the
        per-call cost at the price of device memory.
    is_prefetch_enabled : bool
        Whether iteration decodes ahead in a background thread.
    prefetch_queue_size : int
        Maximum number of frames decoded ahead when prefetching.
    """

    device: Device = field(default_factory=Device.detect)
    decode_run_length: int = 16
    is_prefetch_enabled: bool = False
    prefetch_queue_size: int = 2

    def __post_init__(self) -> None:
        """
        Validate the options.

        Raises
        ------
        ValueError
            If any option is out of range.
        """
        if self.decode_run_length < 1:
            raise ValueError(f"decode_run_length must be at least 1, got {self.decode_run_length}")
        if self.prefetch_queue_size < 1:
            raise ValueError(
                f"prefetch_queue_size must be at least 1, got {self.prefetch_queue_size}"
            )
