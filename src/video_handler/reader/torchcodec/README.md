# torchcodec

## Overview

`VideoFrameReader[RGBTensorFrame]` backed by `torchcodec`. Frames stay on the
decode device as `(3, H, W)` uint8 RGB tensors.

## Components

| Component | Description |
|-----------|-------------|
| [core.py](core.py) | `TorchCodecVideoReader` |
| [options.py](options.py) | `TorchCodecReadOptions`: device, `decode_run_length` and prefetch settings |
| [decoder.py](decoder.py) | `TorchCodecDecoder`: `VideoDecoder` with validated metadata and range checks |
| [cursor.py](cursor.py) | `TorchCodecFrameCursor`: iteration decoding `decode_run_length` frames per call |
