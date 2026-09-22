from enum import Enum


class VideoBackend(Enum):
    """
    Library a video is decoded or encoded with.

    Attributes
    ----------
    OPENCV
        OpenCV (`cv2.VideoCapture` / `cv2.VideoWriter`). CPU only, frames are
        `BGRFrame` NumPy arrays of shape ``(H, W, 3)``.
    TORCHCODEC
        `torchcodec` on a torch device. CUDA decodes on NVDEC, frames are
        `RGBTensorFrame` tensors of shape ``(3, H, W)``.
    """

    OPENCV = "opencv"
    TORCHCODEC = "torchcodec"
