from dataclasses import dataclass


@dataclass(frozen=True)
class VideoReadParameters:
    """
    Which frames a reader yields while iterating, independent of the backend.

    Attributes
    ----------
    start_frame : int
        Index of the first frame to yield.
    frame_step : int
        Distance between consecutive yielded frames.
    stop_frame : int | None
        Exclusive upper bound on yielded frame indices; ``None`` iterates to
        the end of the video.
    """

    start_frame: int = 0
    frame_step: int = 1
    stop_frame: int | None = None

    def __post_init__(self) -> None:
        """
        Validate the parameters.

        Raises
        ------
        ValueError
            If any parameter is out of range.
        """
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, got {self.start_frame}")
        if self.frame_step < 1:
            raise ValueError(f"frame_step must be at least 1, got {self.frame_step}")
        if self.stop_frame is not None and self.stop_frame < self.start_frame:
            raise ValueError(
                f"stop_frame ({self.stop_frame}) must not be less than "
                f"start_frame ({self.start_frame})"
            )

    def is_within_range(self, frame_index: int, frame_count: int) -> bool:
        """
        Return whether a frame index lies inside the iteration range.

        Parameters
        ----------
        frame_index : int
            Candidate frame index.
        frame_count : int
            Frame count of the video; a non-positive value means unknown, in
            which case only `stop_frame` bounds the range.

        Returns
        -------
        bool
            True when the index is before both the end of the video and
            `stop_frame`.
        """
        is_before_video_end = frame_count <= 0 or frame_index < frame_count
        is_before_stop_frame = self.stop_frame is None or frame_index < self.stop_frame
        return is_before_video_end and is_before_stop_frame

    def count_frames(self, frame_count: int) -> int:
        """
        Return how many frames a full iteration yields.

        Parameters
        ----------
        frame_count : int
            Frame count of the video.

        Returns
        -------
        int
            Number of yielded frames, assuming `frame_count` is accurate.
        """
        end_frame = frame_count if self.stop_frame is None else min(frame_count, self.stop_frame)
        return len(range(self.start_frame, end_frame, self.frame_step))
