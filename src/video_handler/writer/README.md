# writer

## Overview

Sequential frame encoding. Both writers implement `VideoFrameWriter`, open the
output on the first `write` once the frame size is known, reject later frames
of a different size, and create missing parent directories.

## Components

| Component | Description |
|-----------|-------------|
| [protocol.py](protocol.py) | `VideoFrameWriter`: the surface every writer implements |
| [parameters.py](parameters.py) | `VideoWriteParameters`: output `fps` |
| [opencv/](opencv/README.md) | `OpenCVVideoWriter` |
| [torchcodec/](torchcodec/README.md) | `TorchCodecVideoWriter` |
