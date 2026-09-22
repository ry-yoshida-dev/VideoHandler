from pathlib import Path

import numpy as np
import pytest
import torch
from torch_modules import Device

from video_handler import (
    FrameConverter,
    OpenCVVideoWriter,
    OpenCVWriteOptions,
    TorchCodecVideoWriter,
    TorchCodecWriteOptions,
    VideoBackend,
    VideoWriteParameters,
)

from .conftest import build_factory, decode_frame_index, encode_frame_index, output_path_in

WRITTEN_FRAME_COUNT: int = 12


class TestWriters:
    def test_opencv_writer_round_trip(self, tmp_path: Path, backend: VideoBackend) -> None:
        output_path = output_path_in(tmp_path, "opencv.mp4")
        with OpenCVVideoWriter(output_path, VideoWriteParameters(fps=15.0)) as writer:
            for frame_index in range(WRITTEN_FRAME_COUNT):
                writer.write(encode_frame_index(frame_index))
            assert writer.written_frame_count == WRITTEN_FRAME_COUNT
        with build_factory(backend).build(output_path) as reader:
            assert reader.metadata.fps == pytest.approx(15.0)
            assert [decode_frame_index(frame) for frame in reader] == list(
                range(WRITTEN_FRAME_COUNT)
            )

    def test_torchcodec_writer_round_trip(self, tmp_path: Path, backend: VideoBackend) -> None:
        output_path = output_path_in(tmp_path, "torchcodec.mp4")
        options = TorchCodecWriteOptions(device=Device.CPU, crf=0, batch_size=5)
        with TorchCodecVideoWriter(output_path, VideoWriteParameters(fps=15.0), options) as writer:
            for frame_index in range(WRITTEN_FRAME_COUNT):
                writer.write(FrameConverter.bgr_to_rgb_tensor(encode_frame_index(frame_index)))
        with build_factory(backend).build(output_path) as reader:
            assert reader.metadata.frame_count == WRITTEN_FRAME_COUNT
            assert [decode_frame_index(frame) for frame in reader] == list(
                range(WRITTEN_FRAME_COUNT)
            )

    def test_opencv_writer_rejects_size_change(self, tmp_path: Path) -> None:
        with OpenCVVideoWriter(output_path_in(tmp_path, "size.mp4")) as writer:
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            with pytest.raises(ValueError):
                writer.write(np.zeros((32, 64, 3), dtype=np.uint8))

    def test_torchcodec_writer_rejects_size_change(self, tmp_path: Path) -> None:
        options = TorchCodecWriteOptions(device=Device.CPU)
        with TorchCodecVideoWriter(output_path_in(tmp_path, "size.mp4"), options=options) as writer:
            writer.write(torch.zeros((3, 48, 64), dtype=torch.uint8))
            with pytest.raises(ValueError):
                writer.write(torch.zeros((3, 32, 64), dtype=torch.uint8))

    def test_opencv_frame_label_leaves_input_untouched(self, tmp_path: Path) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        options = OpenCVWriteOptions(is_frame_label_enabled=True)
        with OpenCVVideoWriter(output_path_in(tmp_path, "label.mp4"), options=options) as writer:
            writer.write(frame)
        assert not frame.any()
