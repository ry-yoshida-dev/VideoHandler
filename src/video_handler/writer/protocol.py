from types import TracebackType
from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class VideoFrameWriter[FrameT](Protocol):
    """
    Sequential frame encoding into one video file.

    The encoder is created on the first `write`, once the frame size is known,
    and every later frame must have that same size.
    """

    @property
    def output_path(self) -> str:
        """
        Return the destination path.

        Returns
        -------
        str
            Path the video is written to.
        """
        ...

    @property
    def written_frame_count(self) -> int:
        """
        Return how many frames have been written.

        Returns
        -------
        int
            Number of frames accepted by `write`.
        """
        ...

    def write(self, frame: FrameT) -> None:
        """
        Append one frame to the video.

        Parameters
        ----------
        frame : FrameT
            Frame to encode.
        """
        ...

    def release(self) -> None:
        """
        Flush pending frames and finalize the file.
        """
        ...

    def __enter__(self) -> Self:
        """
        Enter a context that releases the writer on exit.

        Returns
        -------
        Self
            This writer.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the writer.
        """
        ...
