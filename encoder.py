# TODO: Put your implementation of the encoder here
from pickletools import uint8
import numpy as np
from scipy.fftpack import dct
import quantization as quant
import time


def encoder(yuv_vid):
    '''
        Encodes the video and returns a bitstream
    '''
    blockData = divide_in_blocks(yuv_vid, (32,64))
    dctBlocks = applyDCT(blockData)
    quantizedBlocks = quantization(dctBlocks)
    bitstream = zigzag(quantizedBlocks)
    return bitstream


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
    bitstream = np.empty(totalSize, dtype=np.int32)
    cursor = 0

    zigzagIndices = {}

    for component in ["Y", "U", "V"]:
        
        for key, block in blocks[component].items():
            
            blockSize = block.shape
            pixelCount = blockSize[0] * blockSize[1]

            #If block transform indices haven't been calculated yet
            if blockSize not in zigzagIndices:

                transformArray = np.empty(pixelCount, dtype='uint32, uint32')
                i = 0
                
                #Horizontal case
                for x in range(blockSize[0] - 1):
                    curx = x
                    cury = 0
                    while curx >= 0 and cury < blockSize[1]:
                        transformArray[i] = (curx, cury)
                        i += 1
                        curx -= 1
                        cury += 1

                #Vertical case
                for y in range(blockSize[1]):
                    cury = y
                    curx = blockSize[0] - 1
                    while curx >= 0 and cury < blockSize[1]:
                        transformArray[i] = (curx, cury)
                        i += 1
                        curx -= 1
                        cury += 1

                #Add transform array to known zigzag sequences
                zigzagIndices[blockSize] = transformArray

            #Iterate over all pixels inside a single block
            for n in range(pixelCount):

                #Obtain the zigzag index
                index = zigzagIndices[blockSize][n]

                #Write the pixel to the zigzag destination
                bitstream[cursor + n] = block[index[0], index[1]]

            cursor += pixelCount

    return bitstream