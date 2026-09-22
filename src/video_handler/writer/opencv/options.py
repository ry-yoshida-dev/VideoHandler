from dataclasses import dataclass

from .codec import VideoCodec


@dataclass(frozen=True)
class OpenCVWriteOptions:
    """
    Settings specific to `OpenCVVideoWriter`.

    Attributes
    ----------
    codec : VideoCodec
        FourCC codec.
    is_frame_label_enabled : bool
        Whether ``Frame: {n}`` is drawn onto every written frame.
    frame_label_start : int
        Number drawn on the first frame.
    frame_label_step : int
        Increment of the drawn number per frame.
    """

    codec: VideoCodec = VideoCodec.MP4V
    is_frame_label_enabled: bool = False
    frame_label_start: int = 0
    frame_label_step: int = 1

    def __post_init__(self) -> None:
        """
        Validate the options.

        Raises
        ------
        ValueError
            If any option is out of range.
        """
        if self.frame_label_start < 0:
            raise ValueError(
                f"frame_label_start must be non-negative, got {self.frame_label_start}"
            )
        if self.frame_label_step < 1:
            raise ValueError(f"frame_label_step must be at least 1, got {self.frame_label_step}")
