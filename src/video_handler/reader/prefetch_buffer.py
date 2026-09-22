import queue
import threading

from ..frame import IndexedFrame
from .frame_source import FrameSource
from .stream_end import StreamEnd

RELEASE_JOIN_INTERVAL_SECONDS: float = 0.05


class PrefetchBuffer[FrameT]:
    """
    Decode frames from a `FrameSource` ahead of consumption in a background thread.

    The producer thread owns the wrapped source exclusively, so the source
    must not be touched by anyone else while the buffer runs. An exception
    raised by the source is re-raised to the consumer as a `RuntimeError`.
    """

    def __init__(self, source: FrameSource[FrameT], queue_size: int) -> None:
        """
        Start prefetching.

        Parameters
        ----------
        source : FrameSource[FrameT]
            Source to decode from; ownership passes to the buffer.
        queue_size : int
            Maximum number of decoded frames held ahead of consumption.

        Raises
        ------
        ValueError
            If `queue_size` is not positive.
        """
        if queue_size < 1:
            raise ValueError(f"queue_size must be at least 1, got {queue_size}")
        self._source: FrameSource[FrameT] = source
        self._queue: queue.Queue[IndexedFrame[FrameT] | StreamEnd] = queue.Queue(maxsize=queue_size)
        self._stop_event: threading.Event = threading.Event()
        self._producer_error: Exception | None = None
        self._lookahead: IndexedFrame[FrameT] | StreamEnd | None = None
        self._is_ended: bool = False
        self._thread: threading.Thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        """
        Decode frames into the queue until the source ends or the buffer stops.
        """
        try:
            while not self._stop_event.is_set():
                item = self._source.next_frame()
                if item is None:
                    break
                self._queue.put(item)
        except Exception as error:
            self._producer_error = error
        finally:
            self._queue.put(StreamEnd.END)

    def _peek(self) -> IndexedFrame[FrameT] | StreamEnd:
        """
        Return the next queued item without consuming it, waiting if necessary.

        Returns
        -------
        IndexedFrame[FrameT] | StreamEnd
            The next frame or the end sentinel.
        """
        if self._lookahead is None:
            self._lookahead = self._queue.get()
        return self._lookahead

    @property
    def is_exhausted(self) -> bool:
        """
        Return whether no frame is left, waiting for the producer if necessary.

        Returns
        -------
        bool
            True once `next_frame` would return None.
        """
        return self._is_ended or self._peek() is StreamEnd.END

    def next_frame(self) -> IndexedFrame[FrameT] | None:
        """
        Return the next prefetched frame.

        Returns
        -------
        IndexedFrame[FrameT] | None
            The next frame, or None at the end of the stream.

        Raises
        ------
        RuntimeError
            If the source raised while prefetching.
        """
        if self._is_ended:
            return None
        item = self._peek()
        self._lookahead = None
        match item:
            case StreamEnd.END:
                self._is_ended = True
                if self._producer_error is not None:
                    raise RuntimeError("Frame prefetching failed") from self._producer_error
                return None
            case IndexedFrame():
                return item

    def skip_frame(self) -> int | None:
        """
        Consume and discard the next prefetched frame.

        Returns
        -------
        int | None
            Index of the skipped frame, or None at the end of the stream.
        """
        item = self.next_frame()
        return None if item is None else item.frame_index

    def release(self) -> None:
        """
        Stop the producer thread and release the source.
        """
        self._stop_event.set()
        while self._thread.is_alive():
            while not self._queue.empty():
                self._queue.get_nowait()
            self._thread.join(timeout=RELEASE_JOIN_INTERVAL_SECONDS)
        self._source.release()
        self._is_ended = True
