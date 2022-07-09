# TODO: Put your implementation of the encoder here
from distutils.ccompiler import gen_lib_options
import numpy as np
from scipy.fftpack import idct
from bitbuffer import BitReader
from huffman import HuffmanDecoder
import quantization as quant
from utils import DCPredictor
import zigzag as zz
from header import Header



def decoder(bitstream):

    print("Decoding")

    (header, blocks) = entropyDecompress(bitstream)
    print("Decompressed")

    quantizedBlocks = dequantization(blocks)
    print("Dequantized data")

    idctBlocks = applyIDCT(quantizedBlocks)
    print("Applied IDCT")

    result = reassembleFromBlocks(idctBlocks, header)
    print("Reassembled image")

    return result


def reassembleFromBlocks(blocks, header):

    video = {
        "Y": np.ndarray((header.frameCount, header.lumaSize[1], header.lumaSize[0]), dtype="uint8"),
        "U": np.ndarray((header.frameCount, header.chromaSize[1], header.chromaSize[0]), dtype="uint8"),
        "V": np.ndarray((header.frameCount, header.chromaSize[1], header.chromaSize[0]), dtype="uint8")
    }

    for component in video:

        luma = component == "Y"
        frameSize = header.lumaSize if luma else header.chromaSize

        for block, blockData in blocks[component].items():

            frame = block[0]
            x = block[1]
            y = block[2]

            video[component][frame, y:y+blockData.shape[0], x:x+blockData.shape[1]] = np.clip(np.round(blockData[:min(blockData.shape[0], frameSize[1] - y), :min(blockData.shape[1], frameSize[0] - x)]), 0, 255)

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


def inplaceDecompress(block, bitReader, dcPred, dcDecoder, acDecoder, dezigzag):

    coeffBaseDifference = [0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767]
    entropyPositiveBase = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    dcDelta = 0
    dcMagnitude = dcDecoder.read(bitReader)

    if dcMagnitude != 0:

        dcOffset = bitReader.read(dcMagnitude)
        dcDelta = dcOffset if dcOffset >= entropyPositiveBase[dcMagnitude] else coeffBaseDifference[dcMagnitude] + dcOffset

    dc = dcDelta + dcPred.prediction
    dcPred.prediction = dc

    block[0] = dc

    acIndex = 1
    totalCoeffs = len(block)

    while acIndex < totalCoeffs:

        value = acDecoder.read(bitReader)

        #Check if we had EOB
        if value == 0x4F:
            return

        zeroes = value & 0xF
        sign = value & 0x40

        #Check whether we have an extended sequence
        if value & 0x80:

            extValue = acDecoder.read(bitReader)
            value = extValue | (((value >> 4) & 0x3) << 8)

        #If not, extend value to 8 bits
        else:

            value >>= 4
            value &= 0x3

        if sign:
            value = -value

        block[dezigzag[acIndex]] = value
        acIndex += zeroes + 1



def readStreamHeader(bitReader: BitReader):

    s = bitReader.read(8)
    v = bitReader.read(8)
    c = bitReader.read(8)
    version = bitReader.read(8)

    if s != ord('S') or v != ord('V') or c != ord('C') or version != 0:
        raise RuntimeError("Bad SVC file")

    gopSize = bitReader.read(8)
    lumaSizeW = bitReader.read(32)
    lumaSizeH = bitReader.read(32)
    chromaSizeW = bitReader.read(32)
    chromaSizeH = bitReader.read(32)
    frameCount = bitReader.read(32)

    header = Header((lumaSizeW, lumaSizeH), (chromaSizeW, chromaSizeH), frameCount, gopSize)

    return header



def entropyDecompress(bitstream):

    bitReader = BitReader(bitstream)
    header = readStreamHeader(bitReader)

    blocks = { "Y": {}, "U": {}, "V": {}}
    dezigzagTransforms = {}

    # For simplicity reasons, we assume blocks are in descending order towards the border
    blockLumaEndX = ((header.lumaSize[0] + 7) // 8) * 8
    blockLumaEndY = ((header.lumaSize[1] + 7) // 8) * 8
    blockChromaEndX = ((header.chromaSize[0] + 3) // 4) * 4
    blockChromaEndY = ((header.chromaSize[1] + 3) // 4) * 4
    
    for frameID in range(header.frameCount):

        if frameID & (header.gopSize - 1):
            continue

        biDecoder = HuffmanDecoder()
        dcDecoder = HuffmanDecoder()
        acDecoder = HuffmanDecoder()

        bitReader.align()
        biDecoder.deserialize(bitReader)
        dcDecoder.deserialize(bitReader)
        acDecoder.deserialize(bitReader)

        for component in ["Y", "U", "V"]:

            luma = component == "Y"
            baseBlockDim = 8 if luma else 4

            dcPrediction = DCPredictor()

            blockEndX = blockLumaEndX if luma else blockChromaEndX
            blockEndY = blockLumaEndY if luma else blockChromaEndY
            blockPosition = (0, 0)

            while True:

                # Read block information
                sizeCompressed = biDecoder.read(bitReader)

                blockWidth = (baseBlockDim << (sizeCompressed & 0x3)).astype("int32")
                blockHeight = (baseBlockDim << ((sizeCompressed >> 2) & 0x3)).astype("int32")
                newFrame = sizeCompressed & 0x80

                # Setup block information
                blockKey = (frameID, blockPosition[0], blockPosition[1])
                blockSize = (blockWidth, blockHeight)

                totalPixels = blockWidth * blockHeight

                # Create dezigzag transform
                if blockSize not in dezigzagTransforms:
                    dezigzagTransforms[blockSize] = zz.zigzagTransform(blockSize)

                # Create block buffer
                block = np.zeros(totalPixels, dtype = np.float64)

                # Decompress
                inplaceDecompress(block, bitReader, dcPrediction, dcDecoder, acDecoder, dezigzagTransforms[blockSize])

                # Add block to tree
                blocks[component][blockKey] = np.reshape(block, (blockHeight, blockWidth))

                # Set new block position
                blockPosition = (blockPosition[0] + blockWidth, blockPosition[1])

                # Jump to next block line
                if blockPosition[0] >= blockEndX:
                    blockPosition = (0, blockPosition[1] + blockHeight)

                if blockPosition[1] >= blockEndY:
                    break

        print("Decoded frame {}".format(frameID))

    return (header, blocks)