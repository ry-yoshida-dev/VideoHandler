from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2

from ...frame import FrameConverter
from ...types import BGRFrame
from ..parameters import VideoWriteParameters
from .frame_label import FrameLabelOverlay
from .options import OpenCVWriteOptions

logger: logging.Logger = logging.getLogger(__name__)


class OpenCVVideoWriter:
    """
    Encode `BGRFrame` frames into a video file with `cv2.VideoWriter`.

    Implements `VideoFrameWriter[BGRFrame]`.

    Attributes
    ----------
    parameters : VideoWriteParameters
        Output frame rate.
    options : OpenCVWriteOptions
        Codec and frame-label settings.
    """

    def __init__(
        self,
        output_path: str,
        parameters: VideoWriteParameters = VideoWriteParameters(),
        options: OpenCVWriteOptions = OpenCVWriteOptions(),
    ) -> None:
        """
        Prepare the writer; the file is created on the first `write`.

        Parameters
        ----------
        output_path : str
            Destination path; missing parent directories are created.
        parameters : VideoWriteParameters, optional
            Output frame rate.
        options : OpenCVWriteOptions, optional
            Codec and frame-label settings.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._output_path: str = output_path
        self.parameters: VideoWriteParameters = parameters
        self.options: OpenCVWriteOptions = options
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None
        self._written_frame_count: int = 0
        self._frame_label: FrameLabelOverlay | None = (
            FrameLabelOverlay(options.frame_label_start, options.frame_label_step)
            if options.is_frame_label_enabled
            else None
        )

    @property
    def output_path(self) -> str:
        """
        Return the destination path.

        Returns
        -------
        str
            Path the video is written to.
        """
        return self._output_path

    @property
    def written_frame_count(self) -> int:
        """
        Return how many frames have been written.

        Returns
        -------
        int
            Number of frames accepted by `write`.
        """
        return self._written_frame_count

    def _open(self, frame_size: tuple[int, int]) -> cv2.VideoWriter:
        """
        Create the underlying writer for frames of `frame_size`.

        Parameters
        ----------
        frame_size : tuple[int, int]
            Frame size as ``(width, height)``, the order OpenCV expects.

        Returns
        -------
        cv2.VideoWriter
            Opened writer.

        Raises
        ------
        RuntimeError
            If OpenCV cannot open a writer for this path and codec.
        """
        writer = cv2.VideoWriter(
            self._output_path,
            self.options.codec.fourcc,
            self.parameters.fps,
            frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"OpenCV could not open a writer for {self._output_path} "
                f"with codec {self.options.codec.value}"
            )
        return writer

    def write(self, frame: BGRFrame) -> None:
        """
        Append one frame to the video.

        Parameters
        ----------
        frame : BGRFrame
            Frame of shape ``(H, W, 3)``.

        Raises
        ------
        ValueError
            If the frame size differs from the first written frame.
        RuntimeError
            If the writer has been released.
        """
        validated_frame = FrameConverter.as_bgr_frame(frame)
        frame_size = (validated_frame.shape[1], validated_frame.shape[0])
        if self._writer is None:
            if self._frame_size is not None:
                raise RuntimeError("The writer has already been released")
            self._writer = self._open(frame_size)
            self._frame_size = frame_size
        elif frame_size != self._frame_size:
            raise ValueError(
                f"Frame size (width, height) must stay {self._frame_size}, got {frame_size}"
            )
        if self._frame_label is not None:
            validated_frame = self._frame_label.draw(validated_frame)
        self._writer.write(validated_frame)
        self._written_frame_count += 1

    def release(self) -> None:
        """
        Finalize the file. Calling it again has no effect.
        """
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        logger.info("Video written: %s (%d frames)", self._output_path, self._written_frame_count)

    def __enter__(self) -> Self:
        """
        Enter a context that releases the writer on exit.

        Returns
        -------
        Self
            This writer.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the writer.
        """
        self.release()
