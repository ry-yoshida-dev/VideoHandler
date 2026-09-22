import cv2

from ...types import BGRFrame

LABEL_ORIGIN: tuple[int, int] = (30, 40)
LABEL_FONT_SCALE: float = 0.6
LABEL_OUTLINE_COLOR: tuple[int, int, int] = (0, 0, 0)
LABEL_FILL_COLOR: tuple[int, int, int] = (255, 255, 255)
LABEL_OUTLINE_THICKNESS: int = 4
LABEL_FILL_THICKNESS: int = 2


class FrameLabelOverlay:
    """
    Draw an incrementing ``Frame: {n}`` label onto consecutive frames.
    """

    def __init__(self, start: int, step: int) -> None:
        """
        Initialize the counter.

        Parameters
        ----------
        start : int
            Number drawn on the first frame.
        step : int
            Increment per frame.
        """
        self._next_number: int = start
        self._step: int = step

    def draw(self, frame: BGRFrame) -> BGRFrame:
        """
        Return a labeled copy of `frame` and advance the counter.

        Parameters
        ----------
        frame : BGRFrame
            Frame to label; left unmodified.

        Returns
        -------
        BGRFrame
            Labeled copy.
        """
        labeled_frame = frame.copy()
        text = f"Frame: {self._next_number}"
        for color, thickness in (
            (LABEL_OUTLINE_COLOR, LABEL_OUTLINE_THICKNESS),
            (LABEL_FILL_COLOR, LABEL_FILL_THICKNESS),
        ):
            cv2.putText(
                labeled_frame,
                text,
                LABEL_ORIGIN,
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                color,
                thickness,
            )
        self._next_number += self._step
        return labeled_frame
