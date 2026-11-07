# combinator

## Overview

Tile several videos, read with any backend, into one grid video.

## Components

| Component | Description |
|-----------|-------------|
| [core.py](core.py) | `VideoCombinator`: reads, composes and writes |
| [parameters.py](parameters.py) | `VideoCombinatorParameters`: rows, cell size, stop criteria, output limit, padding color |
| [stop_criteria.py](stop_criteria.py) | `StopCriteria`: `SHORTEST_VIDEO_END` / `LONGEST_VIDEO_END` |
| [grid_layout.py](grid_layout.py) | `GridLayout`: row-major placement of frames on a canvas |
| [letterbox.py](letterbox.py) | `LetterboxResizer`: aspect-preserving resize with padding |
