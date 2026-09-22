# tests

## Overview

Pytest suite. A generated sample video encodes each frame's index in its
intensity, so tests verify exactly which frames a reader returned. Reader and
writer tests run against both backends.

## Components

| Component | Description |
|-----------|-------------|
| [conftest.py](conftest.py) | Sample video fixture, backend parametrization and helpers |
| [test_reader_contract.py](test_reader_contract.py) | `VideoFrameReader` behavior for both backends, with and without prefetch |
| [test_read_parameters.py](test_read_parameters.py) | `VideoReadParameters` validation and range logic |
| [test_frame_converter.py](test_frame_converter.py) | `FrameConverter` conversions and validation |
| [test_writers.py](test_writers.py) | Round trips through both writers |
| [test_combinator.py](test_combinator.py) | `VideoCombinator` grid size and stop behavior |
