from dataclasses import dataclass


@dataclass(frozen=True)
class VideoWriteParameters:
    """
    Output settings shared by every writer backend.

    Attributes
    ----------
    fps : float
        Frame rate of the encoded video.
    """

    fps: float = 30.0

    def __post_init__(self) -> None:
        """
        Validate the parameters.

        Raises
        ------
        ValueError
            If `fps` is not positive.
        """
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
