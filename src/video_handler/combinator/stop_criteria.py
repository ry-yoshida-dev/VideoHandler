from enum import Enum


class StopCriteria(Enum):
    """
    When grid composition of several videos stops.

    Attributes
    ----------
    SHORTEST_VIDEO_END
        Stop as soon as any input video ends.
    LONGEST_VIDEO_END
        Continue until every input video ends; finished inputs show a blank cell.
    """

    SHORTEST_VIDEO_END = "shortest_video_end"
    LONGEST_VIDEO_END = "longest_video_end"
