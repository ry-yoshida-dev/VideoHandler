# opencv

## Overview

`VideoFrameReader[BGRFrame]` backed by `cv2.VideoCapture`.

## Components

| Component | Description |
|-----------|-------------|
| [core.py](core.py) | `OpenCVVideoReader` |
| [options.py](options.py) | `OpenCVReadOptions`: `seek_threshold` and prefetch settings |
| [capture.py](capture.py) | `OpenCVCapture`: capture that decodes through short forward gaps instead of seeking |
| [cursor.py](cursor.py) | `OpenCVFrameCursor`: iteration over the selected frames |

## Notes

- A target at most `seek_threshold` frames ahead is reached with `grab()`; anything else seeks.
- OpenCV's frame count can be inaccurate; iteration also stops when a read fails.
