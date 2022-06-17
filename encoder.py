# TODO: Put your implementation of the encoder here
from pickletools import uint8
import numpy as np
import header
from scipy.fftpack import dct
import quantization as quant
import time



def encoder(yuv_vid, header):
    '''
        Encodes the video and returns a bitstream
    '''
    blockData = divide_in_blocks(yuv_vid, (32,64))
    dctBlocks = applyDCT(blockData)
    quantizedBlocks = quantization(dctBlocks)
    bitstream = entropyCompress(quantizedBlocks, header)
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

    return newDict


def applyDCT(blocks):

    for component in ["Y", "U", "V"]:

        for block in blocks[component]:
            blocks[component][block] = dct(dct(blocks[component][block].T.astype("int64"), norm="ortho").T, norm="ortho") / np.sqrt(blocks[component][block].shape[0] * blocks[component][block].shape[1])

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


def writeByte(x, stream):
    stream[0] = x

def writeShort(x, stream):
    stream[0] = x & 0xFF
    stream[1] = x >> 8


def inplaceCompress(block, stream, dcPred):
    '''
        The first element is the DC component, the rest is AC
        For DC, do a prediction-based scheme
        For AC, compress data as follows:
        ESXXZZZZ [XXXXXXXX]
        E: Extended sequence
        S: Sign bit
        X: Difference in 2s complement
        Z: Zero run length
        EOB is encoded as 11001111 == 0xCF
    '''
    dcDelta = block[0] - dcPred
    dcPred = block[0]
    writeShort(dcDelta, stream)

    eob = False
    acIndex = 1
    cursor = 2
    blockSize = len(block)

    while (acIndex < blockSize):
        
        ac = block[acIndex]
        acIndex += 1
        zeroes = -1
        sign = ac < 0
        
        if sign:
            ac = -ac

        #Count number of trailing zeroes
        for i in range(16):

            if acIndex >= blockSize:
                zeroes = i
                break

            if block[acIndex] != 0:
                zeroes = i
                break

            acIndex += 1

        #Continue counting if EOB
        if ac == 0 and zeroes == -1:

            eob = True
            scanIndex = acIndex
            
            while scanIndex < blockSize:

                if block[scanIndex] != 0:
                    eob = False
                    zeroes = 15
                    break

                scanIndex += 1

            if eob:
                writeByte(0xCF, stream[cursor:])
                return cursor + 1

        #We have our zero count now, encode it

        if (abs(ac) <= 3):
            #No extended sequence
            seq = ((ac & 0x3) << 4) | zeroes

            if sign:
                seq |= 0x40

            writeByte(seq, stream[cursor:])
            cursor += 1

        else:
            #Extended sequence, split value
            if (ac > 1023):
                ac = 1023

            j0 = ac & 0xFF
            j1 = ac >> 8

            writeByte(0x80 | (sign << 6) | (j1 << 4) | zeroes, stream[cursor:])
            writeByte(j0, stream[cursor + 1:])
            cursor += 2

    return cursor



        





def entropyCompress(blocks, header):

    '''
        We have to take metadata into account, such as:
        - Block configuration as compressed pair <u8, u8> [2 bytes]

        For our bitstream size, expect the worst by having zero compression
    '''

    totalSize = header.frameCount * (header.lumaSize[0] * header.lumaSize[1] + header.chromaSize[0] * header.chromaSize[1] * 2) * 2

    for component in ["Y", "U", "V"]:
        totalSize += len(blocks[component].keys()) * 2

    bitstream = np.empty(totalSize, dtype=np.uint8)
    cursor = 0

    zigzagIndices = {}

    for component in ["Y", "U", "V"]:

        dcPrediction = 0
        frame = 0
        
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

            #Prepend block information
            bitstream[cursor + 0] = blockSize[0]
            bitstream[cursor + 1] = blockSize[1]
            cursor += 2

            buffer = np.empty(pixelCount, dtype=np.int32)

            #Iterate over all pixels inside a single block
            for n in range(pixelCount):

                #Obtain the zigzag index
                index = zigzagIndices[blockSize][n]

                #Write the pixel to the zigzag destination
                buffer[n] = block[index[0], index[1]]

            #Compress
            compressedSize = inplaceCompress(buffer, bitstream[cursor:], dcPrediction)

            #Move cursor
            cursor += compressedSize

    return bitstream[:cursor]