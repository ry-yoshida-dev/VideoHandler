from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Self

from torch_modules import Device

from ...device_resolver import TorchCodecDeviceResolver
from ...frame import IndexedFrame
from ...metadata import VideoMetadata
from ...types import RGBTensorFrame
from ..frame_source import FrameSource
from ..parameters import VideoReadParameters
from ..prefetch_buffer import PrefetchBuffer
from .cursor import TorchCodecFrameCursor
from .decoder import TorchCodecDecoder
from .options import TorchCodecReadOptions


class TorchCodecVideoReader:
    """
    Read `RGBTensorFrame` frames from a video file with `torchcodec`.

    Implements `VideoFrameReader[RGBTensorFrame]`. Frames are returned as
    decoded, shape ``(3, H, W)`` uint8 RGB on the decode device, with no
    channel flip, device transfer or NumPy conversion. Iteration and
    `read_frame_at` use separate decoders, so random access never disturbs
    iteration.

    Attributes
    ----------
    video_path : str
        Path to the video file.
    parameters : VideoReadParameters
        Frames yielded by iteration.
    options : TorchCodecReadOptions
        Device, run length and prefetch settings.
    device : Device
        Device decoding actually runs on, after resolving `options.device`.
    """

    def __init__(
        self,
        video_path: str,
        parameters: VideoReadParameters = VideoReadParameters(),
        options: TorchCodecReadOptions = TorchCodecReadOptions(),
    ) -> None:
        """
        Open the video.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        parameters : VideoReadParameters, optional
            Frames yielded by iteration, by default every frame.
        options : TorchCodecReadOptions, optional
            Device, run length and prefetch settings, by default the best
            device on this machine.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        RuntimeError
            If `torchcodec` cannot determine the stream metadata.
        """
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"The video file does not exist: {video_path}")
        self.video_path: str = video_path
        self.parameters: VideoReadParameters = parameters
        self.options: TorchCodecReadOptions = options
        self.device: Device = TorchCodecDeviceResolver.resolve(self.options.device)
        self._random_access_decoder: TorchCodecDecoder | None = TorchCodecDecoder(
            video_path, self.device
        )
        self._metadata: VideoMetadata = self._random_access_decoder.metadata
        self._source: FrameSource[RGBTensorFrame] | None = None
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

    def _require_source(self) -> FrameSource[RGBTensorFrame]:
        """
        Return the iteration source, opening it on first use.

        Returns
        -------
        FrameSource[RGBTensorFrame]
            A cursor, wrapped in a prefetch buffer when prefetching is enabled.
        """
        if self._source is not None:
            return self._source
        cursor = TorchCodecFrameCursor(
            decoder=TorchCodecDecoder(self.video_path, self.device),
            parameters=self.parameters,
            decode_run_length=self.options.decode_run_length,
        )
        source: FrameSource[RGBTensorFrame] = cursor
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

    def __next__(self) -> RGBTensorFrame:
        """
        Return the next frame and advance by `frame_step`.

        Returns
        -------
        RGBTensorFrame
            The next frame, shape ``(3, H, W)``, on the decode device.

        Raises
        ------
        StopIteration
            If no frame is left.
        """
        item: IndexedFrame[RGBTensorFrame] | None = self._require_source().next_frame()
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
            Frame count reported by `torchcodec`.
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

    def read_frame_at(self, frame_index: int) -> RGBTensorFrame:
        """
        Return the frame at an arbitrary index without advancing iteration.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index.

        Returns
        -------
        RGBTensorFrame
            The frame at `frame_index`, shape ``(3, H, W)``, on the decode device.

        Raises
        ------
        IndexError
            If the index lies outside the video.
        RuntimeError
            If the reader has been released.
        """
        if self._random_access_decoder is None:
            raise RuntimeError("The reader has already been released")
        return self._random_access_decoder.read_at(frame_index)

    def release(self) -> None:
        """
        Stop prefetching and drop both decoders.
        """
        self.reset()
        self._random_access_decoder = None

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
        return (
            f"TorchCodecVideoReader(video_path={self.video_path}, device={self.device}, "
            f"metadata={self._metadata})"
        )
