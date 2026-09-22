from dataclasses import dataclass, field

from ..backend import VideoBackend
from ..types import VideoFrame
from .opencv import OpenCVReadOptions, OpenCVVideoReader
from .parameters import VideoReadParameters
from .protocol import VideoFrameReader
from .torchcodec import TorchCodecReadOptions, TorchCodecVideoReader


@dataclass(frozen=True)
class VideoReaderFactory:
    """
    Build readers for a backend chosen at runtime.

    Attributes
    ----------
    backend : VideoBackend
        Backend every built reader decodes with.
    opencv_options : OpenCVReadOptions
        Options applied when `backend` is ``VideoBackend.OPENCV``.
    torchcodec_options : TorchCodecReadOptions
        Options applied when `backend` is ``VideoBackend.TORCHCODEC``.
    """

    backend: VideoBackend = VideoBackend.OPENCV
    opencv_options: OpenCVReadOptions = field(default_factory=OpenCVReadOptions)
    torchcodec_options: TorchCodecReadOptions = field(default_factory=TorchCodecReadOptions)

    def build(
        self,
        video_path: str,
        parameters: VideoReadParameters = VideoReadParameters(),
    ) -> VideoFrameReader[VideoFrame]:
        """
        Build a reader for one video file.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        parameters : VideoReadParameters, optional
            Frames yielded by iteration, by default every frame.

        Returns
        -------
        VideoFrameReader[VideoFrame]
            A reader yielding `BGRFrame` for OpenCV and `RGBTensorFrame` for
            torchcodec.
        """
        match self.backend:
            case VideoBackend.OPENCV:
                return OpenCVVideoReader(video_path, parameters, self.opencv_options)
            case VideoBackend.TORCHCODEC:
                return TorchCodecVideoReader(video_path, parameters, self.torchcodec_options)
