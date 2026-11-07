# torchcodec

## Overview

`VideoFrameWriter[RGBTensorFrame]` backed by `torchcodec`'s streaming `Encoder`.

## Components

| Component | Description |
|-----------|-------------|
| [core.py](core.py) | `TorchCodecVideoWriter` |
| [options.py](options.py) | `TorchCodecWriteOptions`: device, FFmpeg codec, pixel format, CRF, preset, batch size |
