from enum import Enum

import cv2


class VideoCodec(Enum):
    """
    FourCC codes accepted by `cv2.VideoWriter`.

    Attributes
    ----------
    MP4V
        MPEG-4 Part 2, common for ``.mp4``.
    AVC1
        H.264 (avc1); may require an FFmpeg-enabled OpenCV build.
    H264
        H.264, alternative FourCC.
    X264
        H.264 (x264), alternative FourCC.
    XVID
        Xvid MPEG-4, common for ``.avi``.
    MJPG
        Motion JPEG, common for ``.avi``.
    DIVX
        DivX MPEG-4.
    FMP4
        FFmpeg MPEG-4.
    VP80
        VP8, common for ``.webm``.
    VP90
        VP9, common for ``.webm``.
    """

    MP4V = "mp4v"
    AVC1 = "avc1"
    H264 = "H264"
    X264 = "X264"
    XVID = "XVID"
    MJPG = "MJPG"
    DIVX = "DIVX"
    FMP4 = "FMP4"
    VP80 = "VP80"
    VP90 = "VP90"

    @property
    def fourcc(self) -> int:
        """
        Return the FourCC integer passed to `cv2.VideoWriter`.

        Returns
        -------
        int
            FourCC code.
        """
        code: str = self.value
        return cv2.VideoWriter.fourcc(code[0], code[1], code[2], code[3])
