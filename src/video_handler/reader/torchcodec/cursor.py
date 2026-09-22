import torch

from ...frame import IndexedFrame
from ...types import RGBTensorFrame
from ..parameters import VideoReadParameters
from .decoder import TorchCodecDecoder


class TorchCodecFrameCursor:
    """
    Walk the frames selected by `VideoReadParameters`, decoding them in runs.

    Implements `FrameSource[RGBTensorFrame]`. Decoding one frame per call pays
    a seek and a fixed per-call cost each time; decoding a run of
    `decode_run_length` frames at once amortizes it.
    """

    def __init__(
        self,
        decoder: TorchCodecDecoder,
        parameters: VideoReadParameters,
        decode_run_length: int,
    ) -> None:
        """
        Initialize the cursor at `parameters.start_frame`.

        Parameters
        ----------
        decoder : TorchCodecDecoder
            Decoder owned exclusively by this cursor.
        parameters : VideoReadParameters
            Frame selection.
        decode_run_length : int
            Frames decoded per call.
        """
        self._decoder: TorchCodecDecoder | None = decoder
        self._parameters: VideoReadParameters = parameters
        self._decode_run_length: int = decode_run_length
        self._next_frame_index: int = parameters.start_frame
        self._run: torch.Tensor | None = None
        self._run_start_frame_index: int = 0

    def _require_decoder(self) -> TorchCodecDecoder:
        """
        Return the decoder.

        Returns
        -------
        TorchCodecDecoder
            The decoder.

        Raises
        ------
        RuntimeError
            If the cursor has been released.
        """
        if self._decoder is None:
            raise RuntimeError("The cursor has already been released")
        return self._decoder

    @property
    def is_exhausted(self) -> bool:
        """
        Return whether the cursor has no frame left.

        Returns
        -------
        bool
            True once the end of the range is reached.
        """
        return not self._parameters.is_within_range(
            self._next_frame_index, self._require_decoder().metadata.frame_count
        )

    def next_frame(self) -> IndexedFrame[RGBTensorFrame] | None:
        """
        Return the next frame, decoding a new run first when needed.

        Returns
        -------
        IndexedFrame[RGBTensorFrame] | None
            The next frame, a view into the current run, or None at the end.
        """
        if self.is_exhausted:
            return None
        item = IndexedFrame(
            frame_index=self._next_frame_index,
            frame=self._frame_from_run(self._next_frame_index),
        )
        self._next_frame_index += self._parameters.frame_step
        return item

    def _frame_from_run(self, frame_index: int) -> RGBTensorFrame:
        """
        Return a frame out of the held run, decoding a run starting there if it is not held.

        Parameters
        ----------
        frame_index : int
            Frame index reached by iteration.

        Returns
        -------
        RGBTensorFrame
            The frame at `frame_index`.
        """
        step = self._parameters.frame_step
        distance = frame_index - self._run_start_frame_index
        run = self._run
        if run is None or distance % step != 0 or not 0 <= distance // step < len(run):
            run = self._decode_run(frame_index)
            distance = 0
        return run[distance // step]

    def _decode_run(self, start_frame_index: int) -> torch.Tensor:
        """
        Decode and hold the run beginning at `start_frame_index`.

        Parameters
        ----------
        start_frame_index : int
            First frame index of the run.

        Returns
        -------
        torch.Tensor
            Batch of at most `decode_run_length` frames, `frame_step` apart,
            never past `stop_frame` or the end of the video.
        """
        step = self._parameters.frame_step
        stop = start_frame_index + step * self._decode_run_length
        if self._parameters.stop_frame is not None:
            stop = min(stop, self._parameters.stop_frame)
        run = self._require_decoder().read_range(start=start_frame_index, stop=stop, step=step)
        self._run = run
        self._run_start_frame_index = start_frame_index
        return run

    def skip_frame(self) -> int | None:
        """
        Advance past the next frame without decoding it.

        Returns
        -------
        int | None
            Index of the skipped frame, or None at the end.
        """
        if self.is_exhausted:
            return None
        skipped_frame_index = self._next_frame_index
        self._next_frame_index += self._parameters.frame_step
        return skipped_frame_index

    def release(self) -> None:
        """
        Drop the decoder and the held run; `torchcodec` has no explicit close.
        """
        self._run = None
        self._decoder = None
