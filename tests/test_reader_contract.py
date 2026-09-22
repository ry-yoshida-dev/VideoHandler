import pytest

from video_handler import VideoBackend, VideoReadParameters

from .conftest import (
    SAMPLE_FPS,
    SAMPLE_FRAME_COUNT,
    SAMPLE_HEIGHT,
    SAMPLE_WIDTH,
    build_factory,
    decode_frame_index,
)


class TestReaderContract:
    def test_metadata(self, sample_video_path: str, backend: VideoBackend) -> None:
        with build_factory(backend).build(sample_video_path) as reader:
            assert reader.metadata.frame_count == SAMPLE_FRAME_COUNT
            assert reader.metadata.width == SAMPLE_WIDTH
            assert reader.metadata.height == SAMPLE_HEIGHT
            assert reader.metadata.fps == pytest.approx(SAMPLE_FPS)
            assert len(reader) == SAMPLE_FRAME_COUNT

    @pytest.mark.parametrize("is_prefetch_enabled", [False, True], ids=["direct", "prefetch"])
    @pytest.mark.parametrize(
        "parameters",
        [
            VideoReadParameters(),
            VideoReadParameters(frame_step=3),
            VideoReadParameters(start_frame=5, frame_step=2, stop_frame=20),
            VideoReadParameters(frame_step=25),
        ],
        ids=["all", "step", "range", "seek"],
    )
    def test_iteration_yields_selected_frames(
        self,
        sample_video_path: str,
        backend: VideoBackend,
        parameters: VideoReadParameters,
        is_prefetch_enabled: bool,
    ) -> None:
        factory = build_factory(backend, is_prefetch_enabled)
        with factory.build(sample_video_path, parameters) as reader:
            indices = [decode_frame_index(frame) for frame in reader]
            assert reader.is_exhausted
        stop_frame = SAMPLE_FRAME_COUNT if parameters.stop_frame is None else parameters.stop_frame
        assert indices == list(range(parameters.start_frame, stop_frame, parameters.frame_step))

    def test_skip_and_last_frame_index(self, sample_video_path: str, backend: VideoBackend) -> None:
        parameters = VideoReadParameters(frame_step=2)
        with build_factory(backend).build(sample_video_path, parameters) as reader:
            assert reader.last_frame_index is None
            reader.skip()
            assert reader.last_frame_index == 0
            assert decode_frame_index(next(reader)) == 2
            assert reader.last_frame_index == 2

    def test_skip_raises_at_end(self, sample_video_path: str, backend: VideoBackend) -> None:
        parameters = VideoReadParameters(start_frame=28)
        with build_factory(backend).build(sample_video_path, parameters) as reader:
            reader.skip()
            reader.skip()
            with pytest.raises(StopIteration):
                reader.skip()

    def test_reset_restarts_iteration(self, sample_video_path: str, backend: VideoBackend) -> None:
        with build_factory(backend).build(sample_video_path) as reader:
            first_pass = [decode_frame_index(frame) for frame in reader]
            assert list(reader) == []
            reader.reset()
            assert reader.last_frame_index is None
            assert [decode_frame_index(frame) for frame in reader] == first_pass

    def test_random_access_keeps_iteration_position(
        self, sample_video_path: str, backend: VideoBackend
    ) -> None:
        with build_factory(backend).build(sample_video_path) as reader:
            assert decode_frame_index(next(reader)) == 0
            assert decode_frame_index(reader.read_frame_at(20)) == 20
            assert decode_frame_index(reader.read_frame_at(3)) == 3
            assert decode_frame_index(next(reader)) == 1

    def test_random_access_out_of_range(
        self, sample_video_path: str, backend: VideoBackend
    ) -> None:
        with build_factory(backend).build(sample_video_path) as reader, pytest.raises(IndexError):
            reader.read_frame_at(SAMPLE_FRAME_COUNT + 10)

    def test_missing_file(self, backend: VideoBackend) -> None:
        with pytest.raises(FileNotFoundError):
            build_factory(backend).build("missing.mp4")
