from dataclasses import dataclass

from .stop_criteria import StopCriteria


@dataclass(frozen=True)
class VideoCombinatorParameters:
    """
    Grid layout and stop settings for `VideoCombinator`.

    Attributes
    ----------
    rows : int
        Number of grid rows; the column count follows from the input count.
    cell_height : int
        Height of one grid cell in pixels.
    cell_width : int
        Width of one grid cell in pixels.
    stop_criteria : StopCriteria
        When composition stops.
    max_output_frames : int | None
        Upper bound on written frames; None writes until `stop_criteria` stops.
    padding_color : tuple[int, int, int]
        BGR color of letterbox padding and blank cells.
    """

    rows: int = 2
    cell_height: int = 1080
    cell_width: int = 1920
    stop_criteria: StopCriteria = StopCriteria.SHORTEST_VIDEO_END
    max_output_frames: int | None = None
    padding_color: tuple[int, int, int] = (114, 114, 114)

    def __post_init__(self) -> None:
        """
        Validate the parameters.

        Raises
        ------
        ValueError
            If any parameter is out of range.
        """
        if self.rows < 1:
            raise ValueError(f"rows must be at least 1, got {self.rows}")
        if self.cell_height < 1 or self.cell_width < 1:
            raise ValueError(
                f"cell size must be positive, got {self.cell_height}x{self.cell_width}"
            )
        if self.max_output_frames is not None and self.max_output_frames < 1:
            raise ValueError(f"max_output_frames must be at least 1, got {self.max_output_frames}")
        if any(not 0 <= channel <= 255 for channel in self.padding_color):
            raise ValueError(
                f"padding_color channels must be in [0, 255], got {self.padding_color}"
            )
