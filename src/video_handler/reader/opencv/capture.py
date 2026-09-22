import cv2

from ...frame import FrameConverter
from ...metadata import VideoMetadata
from ...types import BGRFrame


class OpenCVCapture:
    """
    `cv2.VideoCapture` that tracks its decode position to avoid needless seeks.

    Reading a frame shortly ahead of the current position decodes and
    discards the frames in between, which is faster than seeking; reading
    anywhere else seeks.
    """

    def __init__(self, video_path: str, seek_threshold: int) -> None:
        """
        Open the video.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        seek_threshold : int
            Largest forward distance covered by decoding instead of seeking.

        Raises
        ------
        ValueError
            If OpenCV cannot open the file.
        """
        self._capture: cv2.VideoCapture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open the video file: {video_path}")
        self._seek_threshold: int = seek_threshold
        self._position: int | None = 0

    @property
    def metadata(self) -> VideoMetadata:
        """
        Return the stream properties OpenCV reports.

        Returns
        -------
        VideoMetadata
            Frame count, frame rate and frame size.
        """
        return VideoMetadata(
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read_at(self, frame_index: int) -> BGRFrame | None:
        """
        Decode the frame at `frame_index`.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index.

        Returns
        -------
        BGRFrame | None
            The frame, or None when OpenCV cannot decode it (e.g. past the end).
        """
        if not self._move_to(frame_index):
            return None
        is_read, frame = self._capture.read()
        if not is_read:
            self._position = None
            return None
        self._position = frame_index + 1
        return FrameConverter.as_bgr_frame(frame)

    def _move_to(self, frame_index: int) -> bool:
        """
        Position the capture so the next decoded frame is `frame_index`.

        Parameters
        ----------
        frame_index : int
            Target frame index.

        Returns
        -------
        bool
            False when decoding through to the target hit the end of the stream.
        """
        position = self._position
        if position is not None and 0 <= frame_index - position <= self._seek_threshold:
            for _ in range(frame_index - position):
                if not self._capture.grab():
                    self._position = None
                    return False
            return True
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return True

    def release(self) -> None:
        """
        Release the underlying capture.
        """
        self._capture.release()
