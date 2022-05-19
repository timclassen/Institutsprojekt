# TODO: Put your implementation of the encoder here
import numpy as np
from scipy.fftpack import dct
import quantization as quant


def encoder(yuv_vid):
    '''
        Encodes the video and returns a bitstream
    '''
    blockData = divide_in_blocks(yuv_vid, (32,64))
    dctBlocks = applyDCT(blockData)
    quantizedBlocks = quantization(dctBlocks)
    bitstream = zigzag(quantizedBlocks)
    return quantizedBlocks


def divide_in_blocks(vid, block_size=(64, 64)):

    newDict = {}
    blockSizes = {"Y":block_size, "U":(block_size[0] // (vid["Y"].shape[1] // vid["U"].shape[1]), block_size[1] // (vid["Y"].shape[2] // vid["U"].shape[2])), "V":(block_size[0] // (vid["Y"].shape[1] // vid["V"].shape[1]), block_size[1] // (vid["Y"].shape[2] // vid["V"].shape[2]))}

    for component in vid:

        compBlockSize = blockSizes[component]

        newDict[component] = {}
        frameCount = vid[component].shape[0]
        height = vid[component][0].shape[1]
        width = vid[component][0].shape[0]
        blocksInX = (width + compBlockSize[0] - 1) // compBlockSize[0]
        blocksInY = (height + compBlockSize[1] - 1) // compBlockSize[1]

        for frame in range(0, vid[component].shape[0]):
            for x in range(0, vid[component].shape[1], compBlockSize[0]):
                for y in range(0, vid[component].shape[2], compBlockSize[1]):
                    newDict[component][(frame, x, y)] = vid[component][frame, x:x + compBlockSize[0], y:y + compBlockSize[1]]

    newDict["luma_frame_size"] = (vid["Y"].shape[1],vid["Y"].shape[2])
    newDict["chroma_frame_size"] = (vid["U"].shape[1],vid["U"].shape[2])
    newDict["frames"] = vid["Y"].shape[0]

    return newDict


def applyDCT(blocks):

    for component in ["Y", "U", "V"]:

        for block in blocks[component]:
            blocks[component][block] = dct(dct(blocks[component][block].T, norm="ortho").T, norm="ortho")

    return blocks


def quantization(blocks):
    quantization_matrices = {}

    for component in ["Y", "U", "V"]:
        for key in blocks[component]:
            blockSize = blocks[component][key].shape
            if blockSize not in quantization_matrices:
                quantization_matrices[blockSize] = quant.get_quantization_matrix(blockSize, quant.DefaultQuantizationFunction)
            blocks[component][key] = np.round(blocks[component][key] / quantization_matrices[blockSize])

    return blocks


def zigzag(blocks):

    totalSize = blocks["frames"] * (blocks["luma_frame_size"][0] * blocks["luma_frame_size"][1] + blocks["chroma_frame_size"][0] * blocks["chroma_frame_size"][1] * 2)
    bitstream = np.empty(totalSize)
    i = 0

    for component in ["Y", "U", "V"]:
        
        for key in blocks[component]:

            blockSize = blocks[component][key].shape

            #Horizontal case
            for x in range(blockSize[0] - 1):
                curx = x
                cury = 0
                while curx >= 0 and cury < blockSize[1]:
                    bitstream[i] = blocks[component][key][curx, cury]
                    i += 1
                    curx -= 1
                    cury += 1

            #Vertical case
            for y in range(blockSize[1]):
                cury = y
                curx = blockSize[0] - 1
                while curx >= 0 and cury < blockSize[1]:
                    bitstream[i] = blocks[component][key][curx, cury]
                    i += 1
                    curx -= 1
                    cury += 1

    return bitstream