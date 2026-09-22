"""Frame type aliases shared by every backend."""

import numpy as np
import torch
from numpy.typing import NDArray

type BGRFrame = NDArray[np.uint8]
type RGBTensorFrame = torch.Tensor
type VideoFrame = BGRFrame | RGBTensorFrame
