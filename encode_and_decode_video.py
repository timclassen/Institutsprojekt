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
    '''
    original_video = {
        "Y": original_video["Y"][:2],
        "U": original_video["U"][:2],
        "V": original_video["V"][:2]
    }
    '''
    
    decodedOutPath = "tmp/decodedVid.yuv"

    header = Header()
    header.lumaSize = (original_video["Y"].shape[1], original_video["Y"].shape[2])
    header.chromaSize = (original_video["U"].shape[1], original_video["U"].shape[2])
    header.frameCount = original_video["Y"].shape[0]
    header.lumaPixels = header.lumaSize[0] * header.lumaSize[1]
    header.chromaPixels = header.chromaSize[0] * header.chromaSize[1]

    print("Original size: {}".format(header.frameCount * (header.lumaSize[0] * header.lumaSize[1] + 2 * header.chromaSize[0] * header.chromaSize[1])))

    bitstream = encoder(original_video, header)
    bitstreamSize = sum(len(v) for k, v in bitstream.items())

    print("Encoded")
    print("Compressed size: {}".format(bitstreamSize))

    decoded_video = decoder(bitstream)
    print("Decoded")

    write_yuv_video(decoded_video, decodedOutPath)
    print("Written to file " + decodedOutPath)

    original_video_size = os.path.getsize(yuv_video_path)

    print("Size of original video:", original_video_size)
    print("Size of bitstream:", bitstreamSize)
    print("Compression ratio:", bitstreamSize / original_video_size)
    print("PSNR:", psnr_yuv(original_video, decoded_video))


encode_and_decode_video("data/ArenaOfValor_384x384_60_8bit_420.yuv")