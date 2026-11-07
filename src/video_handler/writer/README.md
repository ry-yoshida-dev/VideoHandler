# writer

## Overview

Sequential frame encoding. Both writers implement `VideoFrameWriter`.

## Components

| Component | Description |
|-----------|-------------|
| [protocol.py](protocol.py) | `VideoFrameWriter`: the surface every writer implements |
| [parameters.py](parameters.py) | `VideoWriteParameters`: output `fps` |
| [opencv/](opencv/README.md) | `OpenCVVideoWriter` |
| [torchcodec/](torchcodec/README.md) | `TorchCodecVideoWriter` |
