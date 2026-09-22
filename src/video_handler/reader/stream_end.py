from enum import Enum


class StreamEnd(Enum):
    """
    Sentinel a prefetch producer enqueues after its last frame.

    Attributes
    ----------
    END
        No frame follows.
    """

    END = "end"
