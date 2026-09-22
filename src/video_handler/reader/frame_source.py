from typing import Protocol

from ..frame import IndexedFrame


class FrameSource[FrameT](Protocol):
    """
    Producer of consecutive frames that a reader or a prefetch buffer consumes.
    """

    @property
    def is_exhausted(self) -> bool:
        """
        Return whether the source has no frame left.

        Returns
        -------
        bool
            True once `next_frame` would return None.
        """
        ...

    def next_frame(self) -> IndexedFrame[FrameT] | None:
        """
        Decode and return the next frame.

        Returns
        -------
        IndexedFrame[FrameT] | None
            The next frame, or None at the end of the stream.
        """
        ...

    def skip_frame(self) -> int | None:
        """
        Advance past the next frame.

        Returns
        -------
        int | None
            Index of the skipped frame, or None at the end of the stream.
        """
        ...

    def release(self) -> None:
        """
        Release the resources the source holds.
        """
        ...
