# video_handler

## Overview

Backend-agnostic video I/O. Every public symbol is re-exported from
[\_\_init\_\_.py](__init__.py).

## Components

| Component | Description |
|-----------|-------------|
| [backend.py](backend.py) | `VideoBackend` enum (`OPENCV` / `TORCHCODEC`) |
| [types.py](types.py) | `BGRFrame`, `RGBTensorFrame` and `VideoFrame` aliases |
| [metadata.py](metadata.py) | `VideoMetadata`: frame count, fps and frame size |
| [device_resolver.py](device_resolver.py) | `TorchCodecDeviceResolver`: falls back to CPU for devices `torchcodec` cannot use |
| [frame/](frame/README.md) | Frame validation, layout conversion and `IndexedFrame` |
| [reader/](reader/README.md) | `VideoFrameReader` protocol, both readers and `VideoReaderFactory` |
| [writer/](writer/README.md) | `VideoFrameWriter` protocol and both writers |
| [combinator/](combinator/README.md) | `VideoCombinator`: grid composition of several videos |
