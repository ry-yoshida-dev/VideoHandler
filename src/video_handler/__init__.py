"""Backend-agnostic video reading and writing over OpenCV and torchcodec."""

from .backend import VideoBackend
from .combinator import StopCriteria, VideoCombinator, VideoCombinatorParameters
from .frame import FrameConverter, IndexedFrame
from .metadata import VideoMetadata
from .reader import (
    OpenCVReadOptions,
    OpenCVVideoReader,
    TorchCodecReadOptions,
    TorchCodecVideoReader,
    VideoFrameReader,
    VideoReaderFactory,
    VideoReadParameters,
)
from .types import BGRFrame, RGBTensorFrame, VideoFrame
from .writer import (
    OpenCVVideoWriter,
    OpenCVWriteOptions,
    TorchCodecVideoWriter,
    TorchCodecWriteOptions,
    VideoCodec,
    VideoFrameWriter,
    VideoWriteParameters,
)

__all__ = [
    "BGRFrame",
    "FrameConverter",
    "IndexedFrame",
    "OpenCVReadOptions",
    "OpenCVVideoReader",
    "OpenCVVideoWriter",
    "OpenCVWriteOptions",
    "RGBTensorFrame",
    "StopCriteria",
    "TorchCodecReadOptions",
    "TorchCodecVideoReader",
    "TorchCodecVideoWriter",
    "TorchCodecWriteOptions",
    "VideoBackend",
    "VideoCodec",
    "VideoCombinator",
    "VideoCombinatorParameters",
    "VideoFrame",
    "VideoFrameReader",
    "VideoFrameWriter",
    "VideoMetadata",
    "VideoReadParameters",
    "VideoReaderFactory",
    "VideoWriteParameters",
]
