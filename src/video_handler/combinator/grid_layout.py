import math

import numpy as np

from ..types import BGRFrame
from .letterbox import LetterboxResizer
from .parameters import VideoCombinatorParameters


class GridLayout:
    """
    Tile one frame per input video into a row-major grid canvas.

    Attributes
    ----------
    rows : int
        Number of grid rows.
    columns : int
        Number of grid columns.
    """

    def __init__(self, cell_count: int, parameters: VideoCombinatorParameters) -> None:
        """
        Compute the grid for `cell_count` inputs.

        Parameters
        ----------
        cell_count : int
            Number of input videos.
        parameters : VideoCombinatorParameters
            Row count, cell size and padding color.

        Raises
        ------
        ValueError
            If `cell_count` is not positive.
        """
        if cell_count < 1:
            raise ValueError(f"cell_count must be at least 1, got {cell_count}")
        self.rows: int = parameters.rows
        self.columns: int = math.ceil(cell_count / parameters.rows)
        self._cell_count: int = cell_count
        self._cell_height: int = parameters.cell_height
        self._cell_width: int = parameters.cell_width
        self._padding_color: tuple[int, int, int] = parameters.padding_color
        self._resizer: LetterboxResizer = LetterboxResizer(
            parameters.cell_height, parameters.cell_width, parameters.padding_color
        )

    def compose(self, frames: list[BGRFrame | None]) -> BGRFrame:
        """
        Place each frame in its cell; a None entry leaves a blank cell.

        Parameters
        ----------
        frames : list[BGRFrame | None]
            One entry per input, in input order.

        Returns
        -------
        BGRFrame
            Canvas of ``(rows * cell_height, columns * cell_width)``.

        Raises
        ------
        ValueError
            If the number of frames differs from the cell count.
        """
        if len(frames) != self._cell_count:
            raise ValueError(f"Expected {self._cell_count} frames, got {len(frames)}")
        canvas = np.full(
            (self.rows * self._cell_height, self.columns * self._cell_width, 3),
            self._padding_color,
            dtype=np.uint8,
        )
        for cell_index, frame in enumerate(frames):
            if frame is None:
                continue
            row, column = divmod(cell_index, self.columns)
            top = row * self._cell_height
            left = column * self._cell_width
            canvas[top : top + self._cell_height, left : left + self._cell_width] = (
                self._resizer.resize(frame)
            )
        return canvas
