# frame

## Overview

Frame-level utilities shared by readers, writers and the combinator.

## Components

| Component | Description |
|-----------|-------------|
| [converter.py](converter.py) | `FrameConverter`: validates frames and converts between BGR `(H, W, 3)` arrays and RGB `(3, H, W)` tensors |
| [indexed_frame.py](indexed_frame.py) | `IndexedFrame`: a frame paired with its index in the video |
