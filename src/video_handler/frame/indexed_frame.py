from dataclasses import dataclass


@dataclass(frozen=True)
class IndexedFrame[FrameT]:
    """
    A decoded frame together with its position in the video.

    Attributes
    ----------
    frame_index : int
        Zero-based index of the frame within its video file.
    frame : FrameT
        Decoded frame.
    """

    frame_index: int
    frame: FrameT
