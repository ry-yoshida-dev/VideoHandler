from collections.abc import Iterator
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from ..metadata import VideoMetadata


@runtime_checkable
class VideoFrameReader[FrameT](Protocol):
    """
    Sequential and random-access frame reading from one video file.

    Every backend implements this surface, so backend-agnostic code never
    depends on which one is behind it. A reader is a single-pass iterator
    over the frames selected by its `VideoReadParameters`; call `reset` to
    iterate again. Random access through `read_frame_at` never disturbs the
    iteration position.
    """

    @property
    def metadata(self) -> VideoMetadata:
        """
        Return the stream properties of the video.

        Returns
        -------
        VideoMetadata
            Frame count, frame rate and frame size.
        """
        ...

    @property
    def is_exhausted(self) -> bool:
        """
        Return whether iteration has no frame left to yield.

        Returns
        -------
        bool
            True once the next `__next__` would raise `StopIteration`.
        """
        ...

    @property
    def last_frame_index(self) -> int | None:
        """
        Return the index of the frame most recently yielded or skipped.

        Returns
        -------
        int | None
            Frame index, or None before the first step.
        """
        ...

    def __iter__(self) -> Iterator[FrameT]:
        """
        Return the reader itself.

        Returns
        -------
        Iterator[FrameT]
            This reader.
        """
        ...

    def __next__(self) -> FrameT:
        """
        Return the next frame and advance by `frame_step`.

        Returns
        -------
        FrameT
            The next frame.

        Raises
        ------
        StopIteration
            If no frame is left.
        """
        ...

    def __len__(self) -> int:
        """
        Return the frame count of the whole video.

        Returns
        -------
        int
            Frame count reported by the container.
        """
        ...

    def skip(self) -> None:
        """
        Advance past the next frame without returning it, decoding it only if
        the backend cannot avoid that.

        Raises
        ------
        StopIteration
            If no frame is left.
        """
        ...

    def reset(self) -> None:
        """
        Rewind iteration to `start_frame`.
        """
        ...

    def read_frame_at(self, frame_index: int) -> FrameT:
        """
        Return the frame at an arbitrary index without advancing iteration.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index.

        Returns
        -------
        FrameT
            The frame at `frame_index`.

        Raises
        ------
        IndexError
            If the index lies outside the video.
        """
        ...

    def release(self) -> None:
        """
        Release every resource the reader holds.
        """
        ...

    def __enter__(self) -> Self:
        """
        Enter a context that releases the reader on exit.

        Returns
        -------
        Self
            This reader.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the reader.
        """
        ...
