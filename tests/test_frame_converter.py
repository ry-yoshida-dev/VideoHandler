import numpy as np
import pytest
import torch

from video_handler import FrameConverter


class TestFrameConverter:
    def test_bgr_round_trip_through_tensor(self) -> None:
        bgr_frame = np.zeros((4, 6, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 10
        bgr_frame[:, :, 2] = 200
        tensor = FrameConverter.bgr_to_rgb_tensor(bgr_frame)
        assert tuple(tensor.shape) == (3, 4, 6)
        assert int(tensor[0, 0, 0]) == 200
        assert int(tensor[2, 0, 0]) == 10
        np.testing.assert_array_equal(FrameConverter.rgb_tensor_to_bgr(tensor), bgr_frame)

    def test_to_bgr_accepts_both_layouts(self) -> None:
        bgr_frame = np.full((4, 6, 3), 7, dtype=np.uint8)
        tensor = torch.full((3, 4, 6), 7, dtype=torch.uint8)
        np.testing.assert_array_equal(FrameConverter.to_bgr(tensor), bgr_frame)
        np.testing.assert_array_equal(FrameConverter.to_bgr(bgr_frame), bgr_frame)

    def test_as_bgr_frame_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeError):
            FrameConverter.as_bgr_frame(np.zeros((4, 6, 3), dtype=np.float32))

    def test_as_bgr_frame_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            FrameConverter.as_bgr_frame(np.zeros((4, 6), dtype=np.uint8))

    def test_as_rgb_tensor_frame_rejects_channel_last_layout(self) -> None:
        with pytest.raises(ValueError):
            FrameConverter.as_rgb_tensor_frame(torch.zeros((4, 6, 3), dtype=torch.uint8))
