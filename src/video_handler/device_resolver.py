import warnings

import torch
from torch_modules import Device


class TorchCodecDeviceResolver:
    """
    Resolve a requested device to one `torchcodec` can decode or encode on.

    Resolution happens before a decoder or encoder is constructed, since
    `torchcodec` reports an unavailable CUDA device as an opaque dispatcher
    error that names neither the device nor the reason.
    """

    @staticmethod
    def resolve(device: Device) -> Device:
        """
        Return the device `torchcodec` can actually run on here.

        Parameters
        ----------
        device : Device
            Requested device.

        Returns
        -------
        Device
            `device` itself for ``Device.CPU`` and for an available CUDA device,
            otherwise ``Device.CPU`` with a warning. ``Device.MPS`` always falls
            back, since `torchcodec` has no Metal code path.
        """
        match device:
            case Device.CPU:
                return device
            case Device.CUDA if torch.cuda.is_available():
                return device
            case _:
                warnings.warn(
                    f"torchcodec cannot run on {device} here, falling back to {Device.CPU}.",
                    stacklevel=3,
                )
                return Device.CPU
