import numpy as np


def zigzagTransform(blockSize):

    transformArray = np.empty(blockSize[0] * blockSize[1], dtype='uint32')
    i = 0
    
    #Horizontal case
    for x in range(blockSize[0] - 1):
        curx = x
        cury = 0
        while curx >= 0 and cury < blockSize[1]:
            transformArray[i] = cury * blockSize[0]  + curx
            i += 1
            curx -= 1
            cury += 1

    #Vertical case
    for y in range(blockSize[1]):
        cury = y
        curx = blockSize[0] - 1
        while curx >= 0 and cury < blockSize[1]:
            transformArray[i] = cury * blockSize[0]  + curx
            i += 1
            curx -= 1
            cury += 1

    return transformArray

def dezigzagTransform(blockSize):

    transformArray = np.empty(blockSize[0] * blockSize[1], dtype='uint32')
    i = 0
    
    #Horizontal case
    for x in range(blockSize[0] - 1):
        curx = x
        cury = 0
        while curx >= 0 and cury < blockSize[1]:
            transformArray[i] = curx * blockSize[1]  + cury
            i += 1
            curx -= 1
            cury += 1

    #Vertical case
    for y in range(blockSize[1]):
        cury = y
        curx = blockSize[0] - 1
        while curx >= 0 and cury < blockSize[1]:
            transformArray[i] = curx * blockSize[1]  + cury
            i += 1
            curx -= 1
            cury += 1

    return transformArray


ZigzagTransformCache = {}


def zigzag(blockSize):

    if blockSize not in ZigzagTransformCache:
        ZigzagTransformCache[blockSize] = zigzagTransform(blockSize)

    return ZigzagTransformCache[blockSize]