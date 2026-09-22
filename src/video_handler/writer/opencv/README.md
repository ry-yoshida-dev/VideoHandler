# opencv

## Overview

`VideoFrameWriter[BGRFrame]` backed by `cv2.VideoWriter`.

## Components

| Component | Description |
|-----------|-------------|
| [core.py](core.py) | `OpenCVVideoWriter` |
| [options.py](options.py) | `OpenCVWriteOptions`: codec and frame-label settings |
| [codec.py](codec.py) | `VideoCodec`: FourCC enum with `.fourcc` |
| [frame_label.py](frame_label.py) | `FrameLabelOverlay`: draws `Frame: {n}` onto a copy of each frame |
