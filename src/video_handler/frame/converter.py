import numpy as np
import torch
from numpy.typing import NDArray
from torch_modules import Device

from ..types import BGRFrame, RGBTensorFrame, VideoFrame

COLOR_CHANNELS: int = 3


class FrameConverter:
    """
    Validate frames and convert them between the OpenCV and torchcodec layouts.

    OpenCV frames are `BGRFrame` arrays of shape ``(H, W, 3)``; torchcodec
    frames are `RGBTensorFrame` tensors of shape ``(3, H, W)``. Both are uint8.
    """

    @staticmethod
    def as_bgr_frame(array: NDArray[np.generic]) -> BGRFrame:
        """
        Validate and narrow a NumPy array to `BGRFrame`.

        Parameters
        ----------
        array : NDArray[np.generic]
            Raw array, typically returned by an OpenCV API.

        Returns
        -------
        BGRFrame
            The same array, narrowed.

        Raises
        ------
        TypeError
            If the dtype is not uint8.
        ValueError
            If the shape is not ``(H, W, 3)``.
        """
        if array.dtype != np.uint8:
            raise TypeError(f"Expected uint8 dtype, got {array.dtype}")
        if array.ndim != 3 or array.shape[2] != COLOR_CHANNELS:
            raise ValueError(f"Expected shape (H, W, {COLOR_CHANNELS}), got {tuple(array.shape)}")
        return array.astype(np.uint8, copy=False)

    @staticmethod
    def as_rgb_tensor_frame(tensor: torch.Tensor) -> RGBTensorFrame:
        """
        Validate a tensor as `RGBTensorFrame`.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to validate.

        Returns
        -------
        RGBTensorFrame
            The same tensor.

        Raises
        ------
        TypeError
            If the dtype is not ``torch.uint8``.
        ValueError
            If the shape is not ``(3, H, W)``.
        """
        if tensor.dtype != torch.uint8:
            raise TypeError(f"Expected torch.uint8 dtype, got {tensor.dtype}")
        if tensor.ndim != 3 or tensor.shape[0] != COLOR_CHANNELS:
            raise ValueError(f"Expected shape ({COLOR_CHANNELS}, H, W), got {tuple(tensor.shape)}")
        return tensor

    @classmethod
    def bgr_to_rgb_tensor(cls, frame: BGRFrame, device: Device = Device.CPU) -> RGBTensorFrame:
        """
        Convert an OpenCV frame to a torchcodec frame.

        Parameters
        ----------
        frame : BGRFrame
            Frame of shape ``(H, W, 3)`` in BGR order.
        device : Device, optional
            Device the returned tensor is placed on, by default ``Device.CPU``.

        Returns
        -------
        RGBTensorFrame
            Contiguous tensor of shape ``(3, H, W)`` in RGB order.
        """
        validated_frame = cls.as_bgr_frame(frame)
        rgb_array = np.ascontiguousarray(validated_frame[:, :, ::-1].transpose(2, 0, 1))
        return torch.as_tensor(rgb_array, device=device.torch_device)

    @classmethod
    def rgb_tensor_to_bgr(cls, frame: RGBTensorFrame) -> BGRFrame:
        """
        Convert a torchcodec frame to an OpenCV frame.

        Parameters
        ----------
        frame : RGBTensorFrame
            Tensor of shape ``(3, H, W)`` in RGB order, on any device.

        Returns
        -------
        BGRFrame
            Contiguous array of shape ``(H, W, 3)`` in BGR order.
        """
        validated_frame = cls.as_rgb_tensor_frame(frame)
        rgb_array: NDArray[np.uint8] = validated_frame.permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(rgb_array[:, :, ::-1])

    @classmethod
    def to_bgr(cls, frame: VideoFrame) -> BGRFrame:
        """
        Convert a frame from either backend to an OpenCV frame.

        Parameters
        ----------
        frame : VideoFrame
            Frame from any backend.

        Returns
        -------
        BGRFrame
            The frame in BGR ``(H, W, 3)`` layout.
        """
        match frame:
            case torch.Tensor():
                return cls.rgb_tensor_to_bgr(frame)
            case _:
                return cls.as_bgr_frame(frame)

    @classmethod
    def to_rgb_tensor(cls, frame: VideoFrame, device: Device = Device.CPU) -> RGBTensorFrame:
        """
        Convert a frame from either backend to a torchcodec frame.

        Parameters
        ----------
        frame : VideoFrame
            Frame from any backend.
        device : Device, optional
            Device the returned tensor is placed on, by default ``Device.CPU``.

        Returns
        -------
        RGBTensorFrame
            The frame in RGB ``(3, H, W)`` layout on `device`.
        """
        match frame:
            case torch.Tensor():
                return cls.as_rgb_tensor_frame(frame).to(device.torch_device)
            case _:
                return cls.bgr_to_rgb_tensor(frame, device)
