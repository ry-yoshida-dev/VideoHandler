# reader

## Overview

Sequential and random-access frame reading. Both readers implement `VideoFrameReader` and can
prefetch frames in a background thread.

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
