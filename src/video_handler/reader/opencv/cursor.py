from ...frame import IndexedFrame
from ...types import BGRFrame
from ..parameters import VideoReadParameters
from .capture import OpenCVCapture


class OpenCVFrameCursor:
    """
    Walk the frames selected by `VideoReadParameters` through its own capture.

    Implements `FrameSource[BGRFrame]`.
    """

    def __init__(
        self,
        capture: OpenCVCapture,
        parameters: VideoReadParameters,
        frame_count: int,
    ) -> None:
        """
        Initialize the cursor at `parameters.start_frame`.

        Parameters
        ----------
        capture : OpenCVCapture
            Capture owned exclusively by this cursor.
        parameters : VideoReadParameters
            Frame selection.
        frame_count : int
            Frame count of the video; non-positive when unknown.
        """
        self._capture: OpenCVCapture = capture
        self._parameters: VideoReadParameters = parameters
        self._frame_count: int = frame_count
        self._next_frame_index: int = parameters.start_frame
        self._is_stream_ended: bool = False

    @property
    def is_exhausted(self) -> bool:
        """
        Return whether the cursor has no frame left.

        Returns
        -------
        bool
            True once the end of the range or of the stream is reached.
        """
        return self._is_stream_ended or not self._parameters.is_within_range(
            self._next_frame_index, self._frame_count
        )

    def next_frame(self) -> IndexedFrame[BGRFrame] | None:
        """
        Decode and return the next frame.

        Returns
        -------
        IndexedFrame[BGRFrame] | None
            The next frame, or None at the end.
        """
        if self.is_exhausted:
            return None
        frame = self._capture.read_at(self._next_frame_index)
        if frame is None:
            self._is_stream_ended = True
            return None
        item = IndexedFrame(frame_index=self._next_frame_index, frame=frame)
        self._next_frame_index += self._parameters.frame_step
        return item

    def skip_frame(self) -> int | None:
        """
        Advance past the next frame without decoding it.

        Returns
        -------
        int | None
            Index of the skipped frame, or None at the end.
        """
        if self.is_exhausted:
            return None
        skipped_frame_index = self._next_frame_index
        self._next_frame_index += self._parameters.frame_step
        return skipped_frame_index

    def release(self) -> None:
        """
        Release the capture.
        """
        self._capture.release()
