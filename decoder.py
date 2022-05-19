# TODO: Put your implementation of the encoder here
import numpy as np
from scipy.fftpack import idct
import quantization as quant


def decoder(bitstream):

    dezigzagBlocks = dezigzag(bitstream)
    quantizedBlocks = dequantization(dezigzagBlocks)
    idctBlocks = applyIDCT(quantizedBlocks)
    result = reassembleFromBlocks(idctBlocks)
    return result


def reassembleFromBlocks(blockDict):

    frames = blockDict["frames"]

    newDict={}
    newDict["Y"]=np.ndarray((frames, blockDict["luma_frame_size"][0], blockDict["luma_frame_size"][1]))
    newDict["U"]=np.ndarray((frames, blockDict["chroma_frame_size"][0], blockDict["chroma_frame_size"][1]))
    newDict["V"]=np.ndarray((frames, blockDict["chroma_frame_size"][0], blockDict["chroma_frame_size"][1]))

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