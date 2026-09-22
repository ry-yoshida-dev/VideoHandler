from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Self

from ...frame import IndexedFrame
from ...metadata import VideoMetadata
from ...types import BGRFrame
from ..frame_source import FrameSource
from ..parameters import VideoReadParameters
from ..prefetch_buffer import PrefetchBuffer
from .capture import OpenCVCapture
from .cursor import OpenCVFrameCursor
from .options import OpenCVReadOptions


class OpenCVVideoReader:
    """
    Read `BGRFrame` frames from a video file with OpenCV.

    Implements `VideoFrameReader[BGRFrame]`. Iteration and `read_frame_at`
    use separate captures, so random access never disturbs iteration, and
    iteration may prefetch in a background thread.

    Attributes
    ----------
    video_path : str
        Path to the video file.
    parameters : VideoReadParameters
        Frames yielded by iteration.
    options : OpenCVReadOptions
        Seek strategy and prefetch settings.
    """

    def __init__(
        self,
        video_path: str,
        parameters: VideoReadParameters = VideoReadParameters(),
        options: OpenCVReadOptions = OpenCVReadOptions(),
    ) -> None:
        """
        Open the video.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        parameters : VideoReadParameters, optional
            Frames yielded by iteration, by default every frame.
        options : OpenCVReadOptions, optional
            Seek strategy and prefetch settings.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If OpenCV cannot open the file.
        """
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"The video file does not exist: {video_path}")
        self.video_path: str = video_path
        self.parameters: VideoReadParameters = parameters
        self.options: OpenCVReadOptions = options
        self._random_access_capture: OpenCVCapture = OpenCVCapture(
            video_path, options.seek_threshold
        )
        self._metadata: VideoMetadata = self._random_access_capture.metadata
        self._source: FrameSource[BGRFrame] | None = None
        self._last_frame_index: int | None = None

    @property
    def metadata(self) -> VideoMetadata:
        """
        Return the stream properties of the video.

        Returns
        -------
        VideoMetadata
            Frame count, frame rate and frame size.
        """
        return self._metadata

    @property
    def is_exhausted(self) -> bool:
        """
        Return whether iteration has no frame left to yield.

        Returns
        -------
        bool
            True once the next `__next__` would raise `StopIteration`.
        """
        if self._source is None:
            return not self.parameters.is_within_range(
                self.parameters.start_frame, self._metadata.frame_count
            )
        return self._source.is_exhausted

    @property
    def last_frame_index(self) -> int | None:
        """
        Return the index of the frame most recently yielded or skipped.

        Returns
        -------
        int | None
            Frame index, or None before the first step.
        """
        return self._last_frame_index

    def _require_source(self) -> FrameSource[BGRFrame]:
        """
        Return the iteration source, opening it on first use.

        Returns
        -------
        FrameSource[BGRFrame]
            A cursor, wrapped in a prefetch buffer when prefetching is enabled.
        """
        if self._source is not None:
            return self._source
        cursor = OpenCVFrameCursor(
            capture=OpenCVCapture(self.video_path, self.options.seek_threshold),
            parameters=self.parameters,
            frame_count=self._metadata.frame_count,
        )
        source: FrameSource[BGRFrame] = cursor
        if self.options.is_prefetch_enabled:
            source = PrefetchBuffer(cursor, self.options.prefetch_queue_size)
        self._source = source
        return source

    def __iter__(self) -> Self:
        """
        Return the reader itself.

        Returns
        -------
        Self
            This reader.
        """
        return self

    def __next__(self) -> BGRFrame:
        """
        Return the next frame and advance by `frame_step`.

        Returns
        -------
        BGRFrame
            The next frame.

        Raises
        ------
        StopIteration
            If no frame is left.
        """
        item: IndexedFrame[BGRFrame] | None = self._require_source().next_frame()
        if item is None:
            raise StopIteration
        self._last_frame_index = item.frame_index
        return item.frame

    def __len__(self) -> int:
        """
        Return the frame count of the whole video.

        Returns
        -------
        int
            Frame count reported by OpenCV.
        """
        return self._metadata.frame_count

    def skip(self) -> None:
        """
        Advance past the next frame, without decoding it unless prefetching.

        Raises
        ------
        StopIteration
            If no frame is left.
        """
        skipped_frame_index = self._require_source().skip_frame()
        if skipped_frame_index is None:
            raise StopIteration
        self._last_frame_index = skipped_frame_index

    def reset(self) -> None:
        """
        Rewind iteration to `start_frame`.
        """
        if self._source is not None:
            self._source.release()
            self._source = None
        self._last_frame_index = None

    def read_frame_at(self, frame_index: int) -> BGRFrame:
        """
        Return the frame at an arbitrary index without advancing iteration.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index.

        Returns
        -------
        BGRFrame
            The frame at `frame_index`.

        Raises
        ------
        IndexError
            If the index is negative or OpenCV cannot decode it.
        """
        if frame_index < 0:
            raise IndexError(f"frame_index must be non-negative, got {frame_index}")
        frame = self._random_access_capture.read_at(frame_index)
        if frame is None:
            raise IndexError(f"Failed to read frame {frame_index} from {self.video_path}")
        return frame

    def release(self) -> None:
        """
        Stop prefetching and release both captures.
        """
        self.reset()
        self._random_access_capture.release()

    def __enter__(self) -> Self:
        """
        Enter a context that releases the reader on exit.

        Returns
        -------
        Self
            This reader.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the reader.
        """
        self.release()

    def __str__(self) -> str:
        return f"OpenCVVideoReader(video_path={self.video_path}, metadata={self._metadata})"
