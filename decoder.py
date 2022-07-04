# TODO: Put your implementation of the encoder here
import numpy as np
from scipy.fftpack import idct
import quantization as quant
import zigzag as zz
from header import Header



def decoder(bitstream):

    header = readStreamHeader(bitstream)
    dezigzagBlocks = entropyDecompress(bitstream, header)
    quantizedBlocks = dequantization(dezigzagBlocks)
    idctBlocks = applyIDCT(quantizedBlocks)
    result = reassembleFromBlocks(idctBlocks, header)
    return result


def reassembleFromBlocks(blockDict, header):

    video = {
        "Y": np.ndarray((header.frameCount, header.lumaSize[0], header.lumaSize[1]), dtype="uint8"),
        "U": np.ndarray((header.frameCount, header.chromaSize[0], header.chromaSize[1]), dtype="uint8"),
        "V": np.ndarray((header.frameCount, header.chromaSize[0], header.chromaSize[1]), dtype="uint8")
    }

    for component in video:

        for block, blockData in blockDict[component].items():

            frame = block[0]
            x = block[1]
            y = block[2]

            video[component][frame, x:x+blockData.shape[0], y:y+blockData.shape[1]] = np.clip(np.round(blockData), 0, 255)

    return video


def applyIDCT(blocks):

    for component in ["Y", "U", "V"]:

        for block in blocks[component]:
            blocks[component][block] = idct(idct(blocks[component][block], axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')

    return blocks


def dequantization(blocks):

    quantization_matrices = {}

    for component in ["Y", "U", "V"]:
        for key in blocks[component]:
            blockSize = blocks[component][key].shape
            if blockSize not in quantization_matrices:
                quantization_matrices[blockSize] = quant.getQuantizationMatrix(blockSize, quant.DefaultQuantizationFunction)
            blocks[component][key] *= quantization_matrices[blockSize]
 
    return blocks


def readByte(stream):
    return stream[0]

def readShort(stream):
    return stream[0] | (stream[1] << 8)

def readInt(stream):
    return stream[0] | (stream[1] << 8) | (stream[2] << 16) | (stream[3] << 24)

    
def inplaceDecompress(dcPred, stream, block, zigzag):
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
    dcDelta = readShort(stream)
    dc = dcDelta + dcPred
    dcPred = dc
    block[0] = dc

    acIndex = 1
    cursor = 2
    totalCoeffs = len(block)

    while acIndex < totalCoeffs:

        value = readByte(stream[cursor:])
        cursor += 1

        #Check if we had EOB
        if value == 0x4F:
            return cursor

        zeroes = value & 0xF
        sign = value & 0x40

        #Check whether we have an extended sequence
        if value & 0x80:

            extValue = readByte(stream[cursor:])
            cursor += 1
            value = extValue | (((value >> 4) & 0x3) << 8)

        #If not, extend value to 8 bits
        else:

            value >>= 4
            value &= 0x3

        if sign:
            value = -value

        block[zigzag[acIndex]] = value
        acIndex += zeroes + 1

    return cursor




def readStreamHeader(bitstream):

    '''
        The header has the following layout:
        1) Luma Width
        2) Luma Height
        3) Chroma Width
        4) Chroma Height
        5) Frame Count
    '''
    headerStream = bitstream["H"]

    header = Header()
    header.lumaSize = (readInt(headerStream), readInt(headerStream[0x4:]))
    header.chromaSize = (readInt(headerStream[0x8:]), readInt(headerStream[0xC:]))
    header.frameCount = readInt(headerStream[0x10:])

    header.lumaPixels = header.lumaSize[0] * header.lumaSize[1]
    header.chromaPixels = header.chromaSize[0] * header.chromaSize[1]

    return header



def entropyDecompress(bitstream, header):

    blocks = {"Y": {}, "U": {}, "V": {}}
    dezigzagTransforms = {}

    for component in ["Y", "U", "V"]:

        subsampled = component == "V" or component == "U"
        baseBlockDim = 4 if subsampled else 8
        dcPrediction = 0
        cursor = 0
        substream = bitstream[component]

        frameID = 0

        while cursor < len(bitstream[component]):

            #Read block information
            sizeCompressed = readByte(substream[cursor:])
            x = readShort(substream[cursor + 1:])
            y = readShort(substream[cursor + 3:])

            cursor += 5

            blockWidth = (baseBlockDim << (sizeCompressed & 0x3)).astype("int32")
            blockHeight = (baseBlockDim << ((sizeCompressed >> 2) & 0x3)).astype("int32")
            newFrame = sizeCompressed & 0x80

            if newFrame:
                frameID += 1

            blockKey = (frameID, x, y)
            blockSize = (blockWidth, blockHeight)

            totalPixels = blockWidth * blockHeight

            if blockSize not in dezigzagTransforms:
                dezigzagTransforms[blockSize] = zz.dezigzagTransform(blockSize)

            buffer = np.zeros(totalPixels, dtype=np.float64)

            compressedSize = inplaceDecompress(dcPrediction, substream[cursor:], buffer, dezigzagTransforms[blockSize])
            cursor += compressedSize

            blocks[component][blockKey] = np.reshape(buffer, (blockWidth, blockHeight))

    return blocks