# TODO: Put your implementation of the encoder here
import numpy as np
from scipy.fftpack import idct
'''
    newDict = {}
    blockSizes = {"Y":block_size, "U":(block_size[0] // (vid["Y"].shape[1] // vid["U"].shape[1]), block_size[1] // (vid["Y"].shape[2] // vid["U"].shape[2])), "V":(block_size[0] // (vid["Y"].shape[1] // vid["V"].shape[1]), block_size[1] // (vid["Y"].shape[2] // vid["V"].shape[2]))}

    print(blockSizes)

    for component in vid:

        print(component)

        compBlockSize = blockSizes[component]

        newDict[component] = {}
        frameCount = vid[component].shape[0]
        width = vid[component][0].shape[0]
        height = vid[component][0].shape[1]
        blocksInX = (width + compBlockSize[0] - 1) // compBlockSize[0]
        blocksInY = (height + compBlockSize[1] - 1) // compBlockSize[1]

        for frame in range(0, vid[compo
(frame, x // compBlockSize[0], y // compBlockSize[1])] = vid[component][frame, x:x + compBlockSize[0], y:y + compBlockSize[1]]

    print(width, height, blocksInX, blocksInY)
  newDict["luma_frame_size"] = (vid["Y"].shape[1],vid["Y"].shape[2])
    newDict["chroma_frame_size"] = (vid["U"].shape[1],vid["U"].shape[2])
'''

def decoder(bitstream):

    idctBlocks = applyIDCT(bitstream)
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