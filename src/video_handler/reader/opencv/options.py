from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCVReadOptions:
    """
    Settings specific to `OpenCVVideoReader`.

    Attributes
    ----------
    seek_threshold : int
        Largest forward distance, in frames, covered by decoding and discarding
        frames; farther targets are reached with a seek instead. Seeking is
        slow for small distances, decoding through is slow for large ones.
    is_prefetch_enabled : bool
        Whether iteration decodes ahead in a background thread.
    prefetch_queue_size : int
        Maximum number of frames decoded ahead when prefetching.
    """

    seek_threshold: int = 16
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
        if self.seek_threshold < 0:
            raise ValueError(f"seek_threshold must be non-negative, got {self.seek_threshold}")
        if self.prefetch_queue_size < 1:
            raise ValueError(
                f"prefetch_queue_size must be at least 1, got {self.prefetch_queue_size}"
            )
