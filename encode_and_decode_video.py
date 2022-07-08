# Please write your code for encoding and decoding the video
from yuv_io import read_yuv_video, write_yuv_video
from encoder import encoder
from decoder import decoder
from bitstream_io import writeBitstream, readBitstream
from psnr import psnr_yuv
import os


def sliceVideo(video, start, frames):

    video = {
        "Y": video["Y"][start:start + frames],
        "U": video["U"][start:start + frames],
        "V": video["V"][start:start + frames]
    }


def encode_and_decode_video(yuvVideoPath):

    svcPath = "tmp/video.svc"
    decodedOutPath = "tmp/decodedVid.yuv"

    originalVideo = read_yuv_video(yuvVideoPath, bit_depth = 8)

    # sliceVideo(originalVideo, 0, 2)

    bitstream = encoder(originalVideo, (64, 64))
    bitstreamSize = bitstream.size
    print("Encoded")

    writeBitstream(svcPath, bitstream)
    print("Written to file " + svcPath)

    bitstream = readBitstream(svcPath)
    decodedVideo = decoder(bitstream)
    print("Decoded")

    write_yuv_video(decodedVideo, decodedOutPath)
    print("Written to file " + decodedOutPath)
    
    originalVideoSize = os.path.getsize(yuvVideoPath)

    print("Size of original video:", originalVideoSize)
    print("Size of bitstream:", bitstreamSize)
    print("Compression ratio:", originalVideoSize / bitstreamSize)
    print("PSNR:", psnr_yuv(originalVideo, decodedVideo))


encode_and_decode_video("data/BlowingBubbles_384x240_50.yuv")