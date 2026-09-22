# VideoHandler

## Overview

VideoHandler (`video_handler`) reads and writes video through one interface
over two backends:

| Backend | Frame type | Layout | Device |
|---------|------------|--------|--------|
| `VideoBackend.OPENCV` | `BGRFrame` (`NDArray[np.uint8]`) | `(H, W, 3)` BGR | CPU |
| `VideoBackend.TORCHCODEC` | `RGBTensorFrame` (`torch.Tensor`) | `(3, H, W)` RGB | CPU or CUDA (NVDEC / NVENC) |

Readers of both backends implement the `VideoFrameReader` protocol and writers
implement `VideoFrameWriter`, so backend-agnostic code picks a backend at
runtime with `VideoReaderFactory`. `FrameConverter` converts frames between the
two layouts, and `VideoCombinator` tiles several videos into one grid video.

For the package layout, see [src/video_handler/README.md](src/video_handler/README.md).

## Installation

```bash
pip install -e .
```

With development tools (pytest, mypy, basedpyright, ruff):

```bash
pip install -e ".[dev]"
```

`torchcodec` loads the system FFmpeg (versions 4 to 9) at the first decode or
encode call, so install it first, e.g. `brew install ffmpeg` on macOS. On
macOS, loading both FFmpeg and OpenCV's bundled copy prints harmless
`objc ... is implemented in both` warnings.

## Examples

### Reading

```python
from video_handler import OpenCVVideoReader, VideoReadParameters

parameters = VideoReadParameters(start_frame=10, frame_step=5, stop_frame=200)

with OpenCVVideoReader("input.mp4", parameters) as reader:
    print(reader.metadata)
    for frame in reader:
        print(reader.last_frame_index, frame.shape)

    frame_100 = reader.read_frame_at(100)
```

Selecting the backend at runtime:

```python
from torch_modules import Device

from video_handler import TorchCodecReadOptions, VideoBackend, VideoReaderFactory

factory = VideoReaderFactory(
    backend=VideoBackend.TORCHCODEC,
    torchcodec_options=TorchCodecReadOptions(device=Device.CUDA),
)

with factory.build("input.mp4") as reader:
    for frame in reader:
        ...
```

A reader is a single-pass iterator; call `reset()` to iterate again.
`read_frame_at` uses its own capture or decoder and never moves iteration.

### Writing

```python
import numpy as np

from video_handler import OpenCVVideoWriter, OpenCVWriteOptions, VideoCodec, VideoWriteParameters

options = OpenCVWriteOptions(codec=VideoCodec.MP4V, is_frame_label_enabled=True)

with OpenCVVideoWriter("output.mp4", VideoWriteParameters(fps=30), options) as writer:
    for _ in range(100):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
```

```python
import torch
from torch_modules import Device

from video_handler import TorchCodecVideoWriter, TorchCodecWriteOptions, VideoWriteParameters

options = TorchCodecWriteOptions(device=Device.CPU, codec="libx264", crf=18)

with TorchCodecVideoWriter("output.mp4", VideoWriteParameters(fps=30), options) as writer:
    for _ in range(100):
        writer.write(torch.zeros((3, 480, 640), dtype=torch.uint8))
```

### Combining

```python
from video_handler import StopCriteria, VideoCombinator, VideoCombinatorParameters

combinator = VideoCombinator(
    video_paths=["a.mp4", "b.mp4", "c.mp4"],
    output_path="grid.mp4",
    parameters=VideoCombinatorParameters(
        rows=2,
        cell_height=480,
        cell_width=640,
        stop_criteria=StopCriteria.LONGEST_VIDEO_END,
    ),
)
written_frame_count = combinator.combine()
```

## Development

```bash
pytest
mypy src tests
basedpyright
ruff check . && ruff format --check .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
