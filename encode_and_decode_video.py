# Please write your code for encoding and decoding the video
from yuv_io import read_yuv_video, write_yuv_video
from encoder import encoder
from decoder import decoder
from bitstream_io import write_bitstream, read_bitstream
from psnr import psnr_yuv
import os


def encode_and_decode_video(yuv_video_path):
    original_video = read_yuv_video(yuv_video_path)
    bitstream = encoder(original_video)
    bitstream_path = write_bitstream("tmp/encodedVid.yuv", bitstream)

    bitstream_from_file = read_bitstream(bitstream_path)
    decoded_video = decoder(bitstream_from_file)
    write_yuv_video(decoded_video, "tmp/decodedVid.yuv")

    original_video_size = os.path.getsize(yuv_video_path)
    bitstream_size = os.path.getsize(bitstream_path)

    print("Size of original video:", original_video_size)
    print("Size of bitstream:", bitstream_size)
    print("Compression ratio:", bitstream_size / original_video_size)
    print("PSNR:", psnr_yuv(original_video, decoded_video))


encode_and_decode_video("/home/staff/classen/Institutsprojekt/data/ArenaOfValor_384x384_60_8bit_420.yuv")