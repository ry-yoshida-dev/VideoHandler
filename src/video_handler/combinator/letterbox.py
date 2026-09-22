import cv2
import numpy as np

from ..frame import FrameConverter
from ..types import BGRFrame


class LetterboxResizer:
    """
    Fit frames into a fixed size, preserving aspect ratio and padding the rest.
    """

    def __init__(self, height: int, width: int, padding_color: tuple[int, int, int]) -> None:
        """
        Initialize the target size.

        Parameters
        ----------
        height : int
            Output height.
        width : int
            Output width.
        padding_color : tuple[int, int, int]
            BGR color of the padding.
        """
        self._height: int = height
        self._width: int = width
        self._padding_color: tuple[int, int, int] = padding_color

    def blank(self) -> BGRFrame:
        """
        Return a frame filled with the padding color.

        Returns
        -------
        BGRFrame
            Frame of the target size.
        """
        return np.full((self._height, self._width, 3), self._padding_color, dtype=np.uint8)

    def resize(self, frame: BGRFrame) -> BGRFrame:
        """
        Scale `frame` to fit the target size and center it on padding.

        Parameters
        ----------
        frame : BGRFrame
            Source frame.

        Returns
        -------
        BGRFrame
            Frame of the target size.
        """
        source_height, source_width = frame.shape[:2]
        if (source_height, source_width) == (self._height, self._width):
            return frame
        ratio = min(self._width / source_width, self._height / source_height)
        resized_width = max(1, int(source_width * ratio))
        resized_height = max(1, int(source_height * ratio))
        resized_frame = FrameConverter.as_bgr_frame(
            cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        )
        canvas = self.blank()
        y_offset = (self._height - resized_height) // 2
        x_offset = (self._width - resized_width) // 2
        canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = (
            resized_frame
        )
        return canvas
