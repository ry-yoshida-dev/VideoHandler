# torchcodec

## Overview

`VideoFrameWriter[RGBTensorFrame]` backed by `torchcodec`'s streaming
`Encoder`. Frames are buffered in batches of `batch_size`, so memory use does
not grow with the video length.

## Components

| Component | Description |
|-----------|-------------|
| [core.py](core.py) | `TorchCodecVideoWriter` |
| [options.py](options.py) | `TorchCodecWriteOptions`: device, FFmpeg codec, pixel format, CRF, preset, batch size |
