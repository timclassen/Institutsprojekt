import numpy as np
import cv2 as cv


baseQuantizationMatrix = np.array([
    [16, 16, 16, 16, 17, 18, 21, 24],
    [16, 16, 16, 16, 17, 19, 22, 25],
    [16, 16, 17, 18, 20, 22, 25, 29],
    [16, 16, 18, 21, 24, 27, 31, 36],
    [17, 17, 20, 24, 30, 35, 41, 47],
    [18, 19, 22, 27, 35, 44, 54, 65],
    [21, 22, 25, 31, 41, 54, 70, 88],
    [24, 25, 29, 36, 47, 65, 88, 115]
], dtype = np.float64)


QuantizationMatrixCache = {}


def quantizationVectorLength(x, y):
    return np.round(max(np.sqrt(x**2 + y**2), 1))

def quantizationScaled(x, y):
    return 1

def quantizationNone(x, y):
    return 1

def quantizationDCOnly(x, y):
    return 1 if x == 0 and y == 0 else 511

def getQuantizationMatrix(blockSize):

    if blockSize not in QuantizationMatrixCache:

        quantizationMatrix = np.empty(blockSize, dtype = np.float64)

        if DefaultQuantizationFunction == quantizationScaled:
            quantizationMatrix = cv.resize(baseQuantizationMatrix, dsize = (blockSize[1], blockSize[0]), interpolation = cv.INTER_LINEAR_EXACT)
        else:

            for y in range(blockSize[1]):
                for x in range(blockSize[0]):
                    quantizationMatrix[x, y] = quantizationMatrix(x, y)
        
        QuantizationMatrixCache[blockSize] = quantizationMatrix

    return QuantizationMatrixCache[blockSize]


DefaultQuantizationFunction = quantizationScaled