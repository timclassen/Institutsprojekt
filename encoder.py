# TODO: Put your implementation of the encoder here
from pickletools import uint8
import numpy as np
import header
from scipy.fftpack import dct
import quantization as quant
import zigzag as zz
import time



def encoder(video, header):
    '''
        Encodes the video and returns a bitstream
    '''
    blockData = divideInBlocks(video, (8, 8))
    dctBlocks = applyDCT(blockData)
    quantizedBlocks = quantization(dctBlocks)
    bitstream = entropyCompress(quantizedBlocks, header)
    return bitstream


def divideInBlocks(vid, blockSize=(64, 64)):

    newDict = {}
    blockSizes = {
        "Y": blockSize,
        "U": (blockSize[0] // (vid["Y"].shape[1] // vid["U"].shape[1]), blockSize[1] // (vid["Y"].shape[2] // vid["U"].shape[2])),
        "V": (blockSize[0] // (vid["Y"].shape[1] // vid["V"].shape[1]), blockSize[1] // (vid["Y"].shape[2] // vid["V"].shape[2]))
    }

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
                    newDict[component][(frame, x, y)] = vid[component][frame, x:x + compBlockSize[0], y:y + compBlockSize[1]].astype('float64')

    return newDict


def applyDCT(blocks):

    for component in ["Y", "U", "V"]:

        for block in blocks[component]:
            blocks[component][block] = dct(dct(blocks[component][block].T, norm="ortho").T, norm="ortho") / np.sqrt(blocks[component][block].shape[0] * blocks[component][block].shape[1])

    return blocks


def quantization(blocks):
    quantization_matrices = {}

    for component in ["Y", "U", "V"]:
        for key in blocks[component]:
            blockSize = blocks[component][key].shape
            if blockSize not in quantization_matrices:
                quantization_matrices[blockSize] = quant.getQuantizationMatrix(blockSize, quant.DefaultQuantizationFunction)
            blocks[component][key] = np.round(blocks[component][key] / quantization_matrices[blockSize])

    return blocks


def writeByte(x, stream):
    stream[0] = x

def writeShort(x, stream):
    stream[0] = x & 0xFF
    stream[1] = x >> 8

def writeInt(x, stream):
    stream[0] = x & 0xFF
    stream[1] = (x >> 8) & 0xFF
    stream[2] = (x >> 16) & 0xFF
    stream[3] = (x >> 24) & 0xFF


def inplaceCompress(block, stream, dcPred, zigzag):
    '''
        The first element is the DC component, the rest is AC
        For DC, do a prediction-based scheme
        For AC, compress data as follows:
        ESXXZZZZ [XXXXXXXX]
        E: Extended sequence
        S: Sign bit
        X: Difference in 2s complement
        Z: Zero run length
        EOB is encoded as 01001111 == 0x4F
    '''
    dcDelta = block[0] - dcPred
    dcPred = block[0]
    writeShort(dcDelta, stream)

    eob = False
    acIndex = 1
    cursor = 2
    blockSize = len(block)

    while (acIndex < blockSize):
        
        ac = block[zigzag[acIndex]]
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

            if block[zigzag[acIndex]] != 0:
                zeroes = i
                break

            acIndex += 1

        #Continue counting if EOB
        if zeroes == -1:

            zeroes = 15
            acIndex -= 1

            if ac == 0:

                eob = True
                scanIndex = acIndex
                
                while scanIndex < blockSize:

                    if block[zigzag[acIndex]] != 0:
                        eob = False
                        break

                    scanIndex += 1

                if eob:
                    writeByte(0x4F, stream[cursor:])
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



        
def createStreamHeader(header):

    '''
        The header has the following layout:
        1) Luma Width
        2) Luma Height
        3) Chroma Width
        4) Chroma Height
        5) Frame Count
    '''
    binaryHeader = np.empty(4 * 5, dtype='uint8')
    writeInt(header.lumaSize[0], binaryHeader)
    writeInt(header.lumaSize[1], binaryHeader[0x4:])
    writeInt(header.chromaSize[0], binaryHeader[0x8:])
    writeInt(header.chromaSize[1], binaryHeader[0xC:])
    writeInt(header.frameCount, binaryHeader[0x10:])

    return binaryHeader





def entropyCompress(blocks, header):

    '''
        We have to take metadata into account, such as:
        - Block configuration as compressed pair <u8, u8> [2 bytes]

        For our bitstream size, expect the worst by having zero compression
    '''

    lumaSize = header.lumaPixels * header.frameCount * 2 + len(blocks["Y"]) * 5
    chromaSize = header.chromaPixels * header.frameCount * 2 + len(blocks["U"]) * 5

    bitstream = {
        "H": createStreamHeader(header)
    }

    zigzagTransforms = {}

    for component in ["Y", "U", "V"]:

        cursor = 0
        dcPrediction = 0
        frame = 0

        subsampled = component == "V" or component == "U"
        baseBlockShift = 3 if subsampled else 4

        streamSize = lumaSize if component == "Y" else chromaSize
        substream = np.empty(lumaSize if component == "Y" else chromaSize, dtype=np.uint8)

        oldFrameID = 0
        
        for key, block in blocks[component].items():
            
            blockSize = block.shape
            pixelCount = blockSize[0] * blockSize[1]

            #If block transform indices haven't been calculated yet
            if blockSize not in zigzagTransforms:

                #Add transform array to known zigzag sequences
                zigzagTransforms[blockSize] = zz.zigzagTransform(blockSize)

            #Prepend block information
            newFrame = oldFrameID != key[0]

            if newFrame:
                oldFrameID += 1

            writeByte((blockSize[0] >> baseBlockShift).bit_length() | ((blockSize[1] >> baseBlockShift).bit_length() << 2) | (newFrame << 7), substream[cursor:])
            writeShort(key[1], substream[cursor + 1:])
            writeShort(key[2], substream[cursor + 3:])

            cursor += 5

            zigzag = zigzagTransforms[blockSize]
            buffer = np.empty(pixelCount, dtype='int16')

            bufferIndex = 0

            for y in range(blockSize[1]):
                for x in range(blockSize[0]):

                    buffer[bufferIndex] = block[x, y]
                    bufferIndex += 1

            #Compress
            compressedSize = inplaceCompress(buffer, substream[cursor:], dcPrediction, zigzag)

            #Move cursor
            cursor += compressedSize

        bitstream[component] = substream[:cursor]

    return bitstream