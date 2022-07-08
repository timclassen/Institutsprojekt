from yuv_io import read_yuv_video, write_yuv_video
import numpy as np


def readBitstream(filePath):
    return np.fromfile(filePath, dtype = np.uint8)


def writeBitstream(filePath, bitstream):

    with open(filePath, "wb") as file:
        file.write(bitstream)