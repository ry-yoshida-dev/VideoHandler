from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Self

import torch
from torch_modules import Device
from torchcodec.encoders import Encoder, VideoStream

from ...device_resolver import TorchCodecDeviceResolver
from ...frame import FrameConverter
from ...types import RGBTensorFrame
from ..parameters import VideoWriteParameters
from .options import TorchCodecWriteOptions

logger: logging.Logger = logging.getLogger(__name__)


class TorchCodecVideoWriter:
    """
    Encode `RGBTensorFrame` frames into a video file with `torchcodec`.

    Implements `VideoFrameWriter[RGBTensorFrame]`. Frames are buffered on the
    encode device and handed to `torchcodec`'s streaming encoder in batches,
    so memory stays bounded by `batch_size` regardless of the video length.

    Attributes
    ----------
    parameters : VideoWriteParameters
        Output frame rate.
    options : TorchCodecWriteOptions
        Device, codec and batching settings.
    device : Device
        Device encoding actually runs on, after resolving `options.device`.
    """

    def __init__(
        self,
        output_path: str,
        parameters: VideoWriteParameters = VideoWriteParameters(),
        options: TorchCodecWriteOptions = TorchCodecWriteOptions(),
    ) -> None:
        """
        Prepare the writer; the file is created on the first `write`.

        Parameters
        ----------
        output_path : str
            Destination path; its extension selects the container and missing
            parent directories are created.
        parameters : VideoWriteParameters, optional
            Output frame rate.
        options : TorchCodecWriteOptions, optional
            Device, codec and batching settings.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._output_path: str = output_path
        self.parameters: VideoWriteParameters = parameters
        self.options: TorchCodecWriteOptions = options
        self.device: Device = TorchCodecDeviceResolver.resolve(options.device)
        self._encoder: Encoder | None = None
        self._stream: VideoStream | None = None
        self._frame_shape: tuple[int, int, int] | None = None
        self._pending_frames: list[RGBTensorFrame] = []
        self._written_frame_count: int = 0

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
            Number of frames accepted by `write`, including buffered ones.
        """
        return self._written_frame_count

    def _open(self, height: int, width: int) -> VideoStream:
        """
        Create the streaming encoder and open the output file.

        Parameters
        ----------
        height : int
            Frame height.
        width : int
            Frame width.

        Returns
        -------
        VideoStream
            Stream frames are added to.
        """
        encoder = Encoder()
        stream = encoder.add_video(
            height=height,
            width=width,
            frame_rate=self.parameters.fps,
            device=self.device.torch_device,
            codec=self.options.codec,
            pixel_format=self.options.pixel_format,
            crf=self.options.crf,
            preset=self.options.preset,
        )
        encoder.open_file(self._output_path)
        self._encoder = encoder
        return stream

    def write(self, frame: RGBTensorFrame) -> None:
        """
        Append one frame to the video.

        Parameters
        ----------
        frame : RGBTensorFrame
            Frame of shape ``(3, H, W)`` on any device.

        Raises
        ------
        ValueError
            If the frame shape differs from the first written frame.
        RuntimeError
            If the writer has been released.
        """
        validated_frame = FrameConverter.as_rgb_tensor_frame(frame)
        frame_shape = (
            int(validated_frame.shape[0]),
            int(validated_frame.shape[1]),
            int(validated_frame.shape[2]),
        )
        if self._stream is None:
            if self._frame_shape is not None:
                raise RuntimeError("The writer has already been released")
            self._stream = self._open(height=frame_shape[1], width=frame_shape[2])
            self._frame_shape = frame_shape
        elif frame_shape != self._frame_shape:
            raise ValueError(f"Frame shape must stay {self._frame_shape}, got {frame_shape}")
        self._pending_frames.append(validated_frame.to(self.device.torch_device))
        self._written_frame_count += 1
        if len(self._pending_frames) >= self.options.batch_size:
            self._flush(self._stream)

    def _flush(self, stream: VideoStream) -> None:
        """
        Hand every buffered frame to the encoder.

        Parameters
        ----------
        stream : VideoStream
            Stream to add the frames to.
        """
        if not self._pending_frames:
            return
        stream.add_frames(torch.stack(self._pending_frames))
        self._pending_frames = []

    def release(self) -> None:
        """
        Flush buffered frames and finalize the file. Calling it again has no effect.
        """
        if self._stream is None or self._encoder is None:
            return
        self._flush(self._stream)
        self._encoder.close()
        self._stream = None
        self._encoder = None
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
