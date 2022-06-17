# TODO: Put your implementation of the encoder here
import numpy as np
from scipy.fftpack import idct
import quantization as quant


def decoder(bitstream, header):

    dezigzagBlocks = dezigzag(bitstream)
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

    

def dezigzag(bistream):



    return blocks