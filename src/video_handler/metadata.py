from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMetadata:
    """
    Stream properties of one video file.

    Attributes
    ----------
    frame_count : int
        Number of frames reported by the container. OpenCV may report ``0``
        or an inaccurate value for some files.
    fps : float
        Average frame rate.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    """

    frame_count: int
    fps: float
    width: int
    height: int
