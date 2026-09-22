import pytest

from video_handler import VideoReadParameters


class TestVideoReadParameters:
    def test_rejects_negative_start_frame(self) -> None:
        with pytest.raises(ValueError):
            VideoReadParameters(start_frame=-1)

    def test_rejects_non_positive_frame_step(self) -> None:
        with pytest.raises(ValueError):
            VideoReadParameters(frame_step=0)

    def test_rejects_stop_frame_before_start_frame(self) -> None:
        with pytest.raises(ValueError):
            VideoReadParameters(start_frame=5, stop_frame=4)

    def test_is_within_range_respects_frame_count_and_stop_frame(self) -> None:
        parameters = VideoReadParameters(stop_frame=10)
        assert parameters.is_within_range(9, frame_count=30)
        assert not parameters.is_within_range(10, frame_count=30)
        assert not parameters.is_within_range(5, frame_count=5)

    def test_is_within_range_ignores_unknown_frame_count(self) -> None:
        assert VideoReadParameters().is_within_range(1000, frame_count=0)

    def test_count_frames(self) -> None:
        assert VideoReadParameters(start_frame=2, frame_step=3).count_frames(30) == 10
        assert VideoReadParameters(frame_step=4, stop_frame=10).count_frames(30) == 3
