# TODO: Put your implementation for writing and reading bitstream files here
from yuv_io import read_yuv_video, write_yuv_video


def read_bitstream(file_path):
    return read_yuv_video(file_path)


def write_bitstream(file_path, bitstream):
    return write_yuv_video(bitstream, file_path)