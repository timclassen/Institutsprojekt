# Please write your code for encoding and decoding the video
from yuv_io import read_yuv_video, write_yuv_video
from encoder import encoder
from decoder import decoder
from header import Header
from bitstream_io import write_bitstream, read_bitstream
from psnr import psnr_yuv
import os


def encode_and_decode_video(yuv_video_path):

    original_video = read_yuv_video(yuv_video_path)
    original_video = {
        "Y": original_video["Y"][:2],
        "U": original_video["U"][:2],
        "V": original_video["V"][:2]
    }

    header = Header()
    header.lumaSize = (original_video["Y"].shape[1], original_video["Y"].shape[2])
    header.chromaSize = (original_video["U"].shape[1], original_video["U"].shape[2])
    header.frameCount = original_video["Y"].shape[0]

    bitstream = encoder(original_video, header)

    # bitstream_path = write_bitstream("tmp/encodedVid.yuv", bitstream)
    print("Encoded")
    #bitstream_from_file = read_bitstream(bitstream_path)
    decoded_video = decoder(bitstream, header)
    # write_yuv_video(decoded_video, "tmp/decodedVid.yuv")
    print("Decoded")

    # original_video_size = os.path.getsize(yuv_video_path)
    # bitstream_size = os.path.getsize(bitstream_path)

    # print("Size of original video:", original_video_size)
    # print("Size of bitstream:", bitstream_size)
    # print("Compression ratio:", bitstream_size / original_video_size)
    print("PSNR:", psnr_yuv(original_video, decoded_video))


encode_and_decode_video("/home/progruppe1/Desktop/Institutsprojekt/data/ArenaOfValor_384x384_60_8bit_420.yuv")