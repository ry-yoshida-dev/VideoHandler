from torch_modules import Device
from torchcodec.decoders import VideoDecoder

from ...metadata import VideoMetadata
from ...types import RGBTensorFrame


class TorchCodecDecoder:
    """
    `torchcodec` `VideoDecoder` with validated metadata and range checks.
    """

    def __init__(self, video_path: str, device: Device) -> None:
        """
        Open the video.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        device : Device
            Device to decode on, already resolved against this machine.

        Raises
        ------
        RuntimeError
            If `torchcodec` cannot determine the frame count, frame rate or
            frame size.
        """
        self._decoder: VideoDecoder = VideoDecoder(video_path, device=device.torch_device)
        stream_metadata = self._decoder.metadata
        frame_count = stream_metadata.num_frames
        fps = stream_metadata.average_fps
        width = stream_metadata.width
        height = stream_metadata.height
        if frame_count is None or fps is None or width is None or height is None:
            raise RuntimeError(
                f"torchcodec could not determine the stream metadata of {video_path}"
            )
        self.metadata: VideoMetadata = VideoMetadata(
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
        )

    def read_at(self, frame_index: int) -> RGBTensorFrame:
        """
        Decode the frame at `frame_index`.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index.

        Returns
        -------
        RGBTensorFrame
            The frame, shape ``(3, H, W)``, on the decode device.

        Raises
        ------
        IndexError
            If the index lies outside the video.
        """
        if not 0 <= frame_index < self.metadata.frame_count:
            raise IndexError(
                f"frame_index must be in [0, {self.metadata.frame_count}), got {frame_index}"
            )
        return self._decoder.get_frame_at(frame_index).data

    def read_range(self, start: int, stop: int, step: int) -> RGBTensorFrame:
        """
        Decode the frames ``start, start + step, ...`` below `stop` in one call.

        Parameters
        ----------
        start : int
            First frame index.
        stop : int
            Exclusive upper bound, clipped to the frame count.
        step : int
            Distance between decoded frames.

        Returns
        -------
        RGBTensorFrame
            Batch of shape ``(N, 3, H, W)`` on the decode device.
        """
        clipped_stop = min(stop, self.metadata.frame_count)
        return self._decoder.get_frames_in_range(start=start, stop=clipped_stop, step=step).data
