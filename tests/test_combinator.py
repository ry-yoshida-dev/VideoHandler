from pathlib import Path

from video_handler import (
    OpenCVVideoReader,
    StopCriteria,
    VideoBackend,
    VideoCombinator,
    VideoCombinatorParameters,
    VideoReadParameters,
)

from .conftest import SAMPLE_FRAME_COUNT, build_factory, output_path_in


class TestVideoCombinator:
    def test_grid_size_and_shortest_stop(
        self, tmp_path: Path, sample_video_path: str, backend: VideoBackend
    ) -> None:
        output_path = output_path_in(tmp_path, "grid.mp4")
        combinator = VideoCombinator(
            video_paths=[sample_video_path] * 3,
            output_path=output_path,
            parameters=VideoCombinatorParameters(rows=2, cell_height=48, cell_width=64),
            reader_factory=build_factory(backend),
        )
        assert combinator.combine() == SAMPLE_FRAME_COUNT
        with OpenCVVideoReader(output_path) as reader:
            assert (reader.metadata.height, reader.metadata.width) == (96, 128)

    def test_longest_stop_and_output_limit(self, tmp_path: Path, sample_video_path: str) -> None:
        longest = VideoCombinator(
            video_paths=[sample_video_path, sample_video_path],
            output_path=output_path_in(tmp_path, "longest.mp4"),
            parameters=VideoCombinatorParameters(
                rows=1,
                cell_height=24,
                cell_width=32,
                stop_criteria=StopCriteria.LONGEST_VIDEO_END,
            ),
            read_parameters=VideoReadParameters(start_frame=10),
        )
        assert longest.combine() == SAMPLE_FRAME_COUNT - 10
        limited = VideoCombinator(
            video_paths=[sample_video_path],
            output_path=output_path_in(tmp_path, "limited.mp4"),
            parameters=VideoCombinatorParameters(
                rows=1, cell_height=24, cell_width=32, max_output_frames=5
            ),
        )
        assert limited.combine() == 5
