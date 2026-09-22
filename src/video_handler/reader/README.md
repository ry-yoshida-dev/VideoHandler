# reader

## Overview

Sequential and random-access frame reading. Both readers implement
`VideoFrameReader` and share the same structure: a backend capture/decoder
for random access, and a separate cursor for iteration that is optionally
wrapped in a `PrefetchBuffer` decoding ahead in a background thread.

## Components

| Component | Description |
|-----------|-------------|
| [protocol.py](protocol.py) | `VideoFrameReader`: the surface every reader implements |
| [parameters.py](parameters.py) | `VideoReadParameters`: `start_frame`, `frame_step`, `stop_frame` |
| [factory.py](factory.py) | `VideoReaderFactory`: builds a reader for a runtime-selected `VideoBackend` |
| [frame_source.py](frame_source.py) | `FrameSource`: protocol of the cursors a reader iterates |
| [prefetch_buffer.py](prefetch_buffer.py) | `PrefetchBuffer`: background-thread prefetch over any `FrameSource` |
| [stream_end.py](stream_end.py) | `StreamEnd`: end-of-stream sentinel of the prefetch queue |
| [opencv/](opencv/README.md) | `OpenCVVideoReader` |
| [torchcodec/](torchcodec/README.md) | `TorchCodecVideoReader` |

## Behavior

- A reader is a single-pass iterator (`__iter__` returns itself); `reset()` rewinds to `start_frame`.
- `skip()` advances without decoding unless prefetching, where the frame is already decoded.
- `read_frame_at()` raises `IndexError` outside the video and never moves iteration.
- `last_frame_index` is the index of the frame last yielded or skipped.
