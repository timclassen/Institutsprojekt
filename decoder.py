# TODO: Put your implementation of the encoder here
import numpy as np
from scipy.fftpack import idct
import quantization as quant


def decoder(bitstream, header):

    dezigzagBlocks = entropyDecompress(bitstream)
    quantizedBlocks = dequantization(dezigzagBlocks)
    idctBlocks = applyIDCT(quantizedBlocks)
    result = reassembleFromBlocks(idctBlocks, header)
    return result


def reassembleFromBlocks(blockDict, header):

    newDict={}
    newDict["Y"]=np.ndarray((header.frameCount, header.lumaSize[0], header.lumaSize[1]))
    newDict["U"]=np.ndarray((header.frameCount, header.chromaSize[0], header.chromaSize[1]))
    newDict["V"]=np.ndarray((header.frameCount, header.chromaSize[0], header.chromaSize[1]))

    for component in newDict:

        for block, blockData in blockDict[component].items():

            frame = block[0]
            x = block[1]
            y = block[2]

            newDict[component][frame, x:x+blockData.shape[0], y:y+blockData.shape[1]] = blockData

    return newDict


def applyIDCT(blocks):

    for component in ["Y", "U", "V"]:

        for block in blocks[component]:
            blocks[component][block] = idct(idct(blocks[component][block].T, norm="ortho").T, norm="ortho")

    return blocks


def dequantization(blocks):
    quantization_matrices = {}

    for component in ["Y", "U", "V"]:
        for key in blocks[component]:
            blockSize = blocks[component][key].shape
            if blockSize not in quantization_matrices:
                quantization_matrices[blockSize] = quant.get_quantization_matrix(blockSize, quant.DefaultQuantizationFunction)
            blocks[component][key] *= quantization_matrices[blockSize]
 
    return blocks


def readShort(stream):
    return stream[0] | (stream[1] << 8)

    
def inplaceDecompress(dcPred, stream, block):
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
    dcDelta = readShort(stream)
    dc = dcDelta + dcPred
    dcPred = dc
    block[0] = dc

    acIndex = 1
    cursor = 2
    totalCoeffs = block.shape[0]

    while acIndex < totalCoeffs:
        
        value = stream[cursor]
        cursor += 1

        #Check if we had EOB
        if value == 0xCF:
            return cursor

        zeroes = value & 0xF
        sign = value & 0x40

        #Check whether we have an extended sequence
        if value & 0x80:

            extValue = stream[cursor]
            cursor += 1
            value = extValue | (((value >> 4) & 0x3) << 8)

        #If not, extend value to 8 bits
        else:

            value >>= 4
            value &= 0x3

        if sign:
            value = -value

        block[acIndex] = value
        acIndex += zeroes + 1

    return cursor







def entropyDecompress(bitstream):

    blocks = {"Y": {}, "U": {}, "V": {}}
    cursor = 0

    for component in ["Y", "U", "V"]:

        subsampled = component == "V" or component == "U"
        dcPrediction = 0

        for f in range(2):

            for i in range(12):

                ix = i * 32

                if subsampled:
                    ix //= 2

                for j in range(6):

                    iy = j * 64

                    if subsampled:
                        iy //= 2

                    blockWidth =  bitstream[cursor + 0]
                    blockHeight = bitstream[cursor + 1]
                    cursor += 2

                    buffer = np.zeros(blockWidth.astype("int32") * blockHeight, dtype=np.int32)
                    cursor += inplaceDecompress(dcPrediction, bitstream[cursor:], buffer)

                    blocks[component][(f, ix, iy)] = np.reshape(buffer, (blockWidth, blockHeight))

    return blocks