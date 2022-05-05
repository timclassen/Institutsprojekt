# TODO: Put your implementation of the encoder here
import numpy as np
from scipy.fftpack import dct 


def encoder(yuv_vid):
    '''
        Encodes the video and returns a bitstream
    '''
    blockData = divide_in_blocks(yuv_vid, (32,64))
    dctBlocks = applyDCT(blockData)
    return dctBlocks


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