from pathlib import Path

import cv2
import numpy as np
import pytest
from torch_modules import Device

from video_handler import (
    FrameConverter,
    OpenCVReadOptions,
    TorchCodecReadOptions,
    VideoBackend,
    VideoFrame,
    VideoReaderFactory,
)

SAMPLE_FRAME_COUNT: int = 30
SAMPLE_WIDTH: int = 64
SAMPLE_HEIGHT: int = 48
SAMPLE_FPS: float = 10.0
INTENSITY_PER_FRAME: int = 8


def encode_frame_index(frame_index: int) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    """
    Build a uniform gray frame whose intensity encodes `frame_index`.

    Parameters
    ----------
    frame_index : int
        Index to encode.

    Returns
    -------
    np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
        BGR frame of the sample size.
    """
    intensity = frame_index * INTENSITY_PER_FRAME
    return np.full((SAMPLE_HEIGHT, SAMPLE_WIDTH, 3), intensity, dtype=np.uint8)


def decode_frame_index(frame: VideoFrame) -> int:
    """
    Recover the index encoded by `encode_frame_index`, tolerating compression noise.

    Parameters
    ----------
    frame : VideoFrame
        Frame from any backend.

    Returns
    -------
    int
        Encoded frame index.
    """
    mean_intensity = float(FrameConverter.to_bgr(frame).mean())
    return round(mean_intensity / INTENSITY_PER_FRAME)


@pytest.fixture(scope="session")
def sample_video_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Write a short video whose frame intensities encode their indices.

    Returns
    -------
    str
        Path to the video.
    """
    video_path = tmp_path_factory.mktemp("videos") / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter.fourcc("m", "p", "4", "v"),
        SAMPLE_FPS,
        (SAMPLE_WIDTH, SAMPLE_HEIGHT),
    )
    for frame_index in range(SAMPLE_FRAME_COUNT):
        writer.write(encode_frame_index(frame_index))
    writer.release()
    return str(video_path)


@pytest.fixture(params=[VideoBackend.OPENCV, VideoBackend.TORCHCODEC], ids=lambda b: b.value)
def backend(request: pytest.FixtureRequest) -> VideoBackend:
    """
    Parametrize a test over every backend.

    Returns
    -------
    VideoBackend
        Backend under test.
    """
    selected_backend: VideoBackend = request.param
    return selected_backend


def build_factory(backend: VideoBackend, is_prefetch_enabled: bool = False) -> VideoReaderFactory:
    """
    Build a CPU reader factory with small decode runs to exercise run boundaries.

    Parameters
    ----------
    backend : VideoBackend
        Backend to build readers for.
    is_prefetch_enabled : bool, optional
        Whether readers prefetch in a background thread.

    Returns
    -------
    VideoReaderFactory
        Configured factory.
    """
    return VideoReaderFactory(
        backend=backend,
        opencv_options=OpenCVReadOptions(is_prefetch_enabled=is_prefetch_enabled),
        torchcodec_options=TorchCodecReadOptions(
            device=Device.CPU,
            decode_run_length=4,
            is_prefetch_enabled=is_prefetch_enabled,
        ),
    )


def output_path_in(directory: Path, name: str) -> str:
    """
    Return a path for a test output file.

    Parameters
    ----------
    directory : Path
        Temporary directory.
    name : str
        File name.

    Returns
    -------
    str
        Path as a string.
    """
    return str(directory / name)
