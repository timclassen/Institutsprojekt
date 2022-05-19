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
])



def quantizationVectorLength(x, y):
    return np.round(max(np.sqrt(x**2 + y**2), 1))

def quantizationScaled(x, y):
    return 1

def get_quantization_matrix(block_size, quant_func):

    if quant_func == quantizationScaled:
        return cv.resize(baseQuantizationMatrix, None, fx=block_size[1] / 8, fy=block_size[0] / 8, interpolation = cv.INTER_LINEAR_EXACT)

    quantization_matrix = np.zeros(block_size)

    for y in range(block_size[0]):
        for x in range(block_size[1]):
            quantization_matrix[y, x] = quant_func(x, y)

    return quantization_matrix


DefaultQuantizationFunction = quantizationScaled