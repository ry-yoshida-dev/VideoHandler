from __future__ import annotations

import logging
from contextlib import ExitStack

from ..frame import FrameConverter
from ..reader import VideoFrameReader, VideoReaderFactory, VideoReadParameters
from ..types import BGRFrame, VideoFrame
from ..writer import OpenCVVideoWriter, OpenCVWriteOptions, VideoWriteParameters
from .grid_layout import GridLayout
from .parameters import VideoCombinatorParameters
from .stop_criteria import StopCriteria

logger: logging.Logger = logging.getLogger(__name__)


class VideoCombinator:
    """
    Combine several videos into one grid-layout video.

    Inputs are decoded with any backend through `VideoReaderFactory`, converted
    to BGR, letterboxed into cells and encoded with `OpenCVVideoWriter`.

    Attributes
    ----------
    video_paths : list[str]
        Input videos, placed in the grid in row-major order.
    output_path : str
        Destination of the combined video.
    parameters : VideoCombinatorParameters
        Grid layout and stop settings.
    reader_factory : VideoReaderFactory
        Builds the reader for each input.
    read_parameters : VideoReadParameters
        Frames read from each input.
    write_parameters : VideoWriteParameters
        Output frame rate.
    write_options : OpenCVWriteOptions
        Output codec and frame-label settings.
    """

    def __init__(
        self,
        video_paths: list[str],
        output_path: str,
        parameters: VideoCombinatorParameters = VideoCombinatorParameters(),
        reader_factory: VideoReaderFactory = VideoReaderFactory(),
        read_parameters: VideoReadParameters = VideoReadParameters(),
        write_parameters: VideoWriteParameters = VideoWriteParameters(),
        write_options: OpenCVWriteOptions = OpenCVWriteOptions(),
    ) -> None:
        """
        Configure the combinator; no file is opened until `combine`.

        Parameters
        ----------
        video_paths : list[str]
            Input videos.
        output_path : str
            Destination of the combined video.
        parameters : VideoCombinatorParameters, optional
            Grid layout and stop settings.
        reader_factory : VideoReaderFactory, optional
            Builds the reader for each input, by default with OpenCV.
        read_parameters : VideoReadParameters, optional
            Frames read from each input, by default every frame.
        write_parameters : VideoWriteParameters, optional
            Output frame rate.
        write_options : OpenCVWriteOptions, optional
            Output codec and frame-label settings.

        Raises
        ------
        ValueError
            If `video_paths` is empty.
        """
        if not video_paths:
            raise ValueError("video_paths must contain at least one path")
        self.video_paths: list[str] = video_paths
        self.output_path: str = output_path
        self.parameters: VideoCombinatorParameters = parameters
        self.reader_factory: VideoReaderFactory = reader_factory
        self.read_parameters: VideoReadParameters = read_parameters
        self.write_parameters: VideoWriteParameters = write_parameters
        self.write_options: OpenCVWriteOptions = write_options
        self._layout: GridLayout = GridLayout(len(video_paths), parameters)

    def combine(self) -> int:
        """
        Compose the inputs and write the result to `output_path`.

        Returns
        -------
        int
            Number of frames written.
        """
        with ExitStack() as stack:
            readers: list[VideoFrameReader[VideoFrame]] = [
                stack.enter_context(self.reader_factory.build(path, self.read_parameters))
                for path in self.video_paths
            ]
            writer = stack.enter_context(
                OpenCVVideoWriter(self.output_path, self.write_parameters, self.write_options)
            )
            while not self._is_output_limit_reached(writer.written_frame_count):
                frames = [self._next_bgr_frame(reader) for reader in readers]
                if self._should_stop(frames):
                    break
                writer.write(self._layout.compose(frames))
            written_frame_count = writer.written_frame_count
        logger.info("Combined %d videos into %s", len(self.video_paths), self.output_path)
        return written_frame_count

    def _is_output_limit_reached(self, written_frame_count: int) -> bool:
        """
        Return whether `max_output_frames` has been reached.

        Parameters
        ----------
        written_frame_count : int
            Frames written so far.

        Returns
        -------
        bool
            True when no more frames may be written.
        """
        max_output_frames = self.parameters.max_output_frames
        return max_output_frames is not None and written_frame_count >= max_output_frames

    def _should_stop(self, frames: list[BGRFrame | None]) -> bool:
        """
        Decide from one batch of frames whether composition ends.

        Parameters
        ----------
        frames : list[BGRFrame | None]
            Next frame of each input, None for a finished input.

        Returns
        -------
        bool
            True when the batch must not be written.
        """
        match self.parameters.stop_criteria:
            case StopCriteria.SHORTEST_VIDEO_END:
                return any(frame is None for frame in frames)
            case StopCriteria.LONGEST_VIDEO_END:
                return all(frame is None for frame in frames)

    @staticmethod
    def _next_bgr_frame(reader: VideoFrameReader[VideoFrame]) -> BGRFrame | None:
        """
        Read the next frame of one input as BGR.

        Parameters
        ----------
        reader : VideoFrameReader[VideoFrame]
            Reader of the input.

        Returns
        -------
        BGRFrame | None
            The frame, or None when the input has ended.
        """
        frame = next(reader, None)
        return None if frame is None else FrameConverter.to_bgr(frame)

    def __str__(self) -> str:
        return (
            f"VideoCombinator(video_paths={self.video_paths}, "
            f"output_path={self.output_path}, parameters={self.parameters})"
        )
