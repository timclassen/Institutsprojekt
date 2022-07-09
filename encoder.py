# TODO: Put your implementation of the encoder here
from base64 import encode
from bz2 import compress
from venv import create
import numpy as np
from bitbuffer import BitBuffer
from scipy.fftpack import dct
from header import Header
from huffman import HuffmanEncoder
import quantization as quant
from utils import *
import zigzag as zz



def encoder(video, baseBlockSize):

    '''
        Encodes the video and returns a bitstream
    '''

    print("Encoding")

    header = Header((video["Y"].shape[2], video["Y"].shape[1]), (video["U"].shape[2], video["U"].shape[1]), video["Y"].shape[0], 4)

    (iFrames, pFrames) = splitFrames(video, header.gopSize)

    
    blockData = divideInBlocks(header, iFrames, baseBlockSize)
    print("Subdivided frame into {} x {} blocks".format(baseBlockSize[0], baseBlockSize[1]))

    dctBlocks = applyDCT(blockData)
    print("Applied DCT")

    quantizedBlocks = quantization(dctBlocks)
    print("Quantized data")

    bitstream = entropyCompress(quantizedBlocks, header)
    print("Compressed")

    return bitstream



def splitFrames(video, gopSize):

    frameCount = video["Y"].shape[0]
    iFrameCount = (frameCount + (gopSize - 1)) // gopSize
    pFrameCount = frameCount - iFrameCount
    
    iFrames = {}
    pFrames = {}

    for component, frames in video.items():

        for i in range(iFrameCount):
            frames[[i, i * gopSize]] = frames[[i * gopSize, i]]

        iFrames[component] = frames[:iFrameCount]
        pFrames[component] = np.sort(frames[iFrameCount:pFrameCount], axis = 0)

    return iFrames, pFrames



def divideInBlocks(header, video, blockSize = (64, 64)):

    subsamplingFactor = (header.lumaSize[0] // header.chromaSize[1], header.lumaSize[1] // header.chromaSize[1])

    blockSizes = {
        "Y": blockSize,
        "U": (blockSize[0] // subsamplingFactor[0], blockSize[1] // subsamplingFactor[1]),
        "V": (blockSize[0] // subsamplingFactor[0], blockSize[1] // subsamplingFactor[1])
    }

    blocks = { "Y": {}, "U": {}, "V": {}}
    frameCount = video["Y"].shape[0]

    for component, frames in video.items():

        luma = component == "Y"
        minBlockSize = 8 if luma else 4

        frameSize = header.lumaSize if luma else header.chromaSize
        width = (frameSize[0] + (minBlockSize - 1)) // minBlockSize * minBlockSize
        height = (frameSize[1] + (minBlockSize - 1)) // minBlockSize * minBlockSize

        baseBlockSize = blockSizes[component]

        for frame in range(frameCount):

            y = 0

            while y < height:

                x = 0
                blockHeight = baseBlockSize[1]

                while y + blockHeight > height:
                    blockHeight >>= 1

                while x < width:

                    blockWidth = baseBlockSize[0]

                    while x + blockWidth > width:
                        blockWidth >>= 1

                    slot = frames[frame, y:y + blockHeight, x:x + blockWidth]

                    blockArray = np.zeros((blockHeight, blockWidth), dtype = np.float64)
                    blockArray[:min(blockHeight, slot.shape[0]), :min(blockWidth, slot.shape[1])] = slot
                    blocks[component][(frame, x, y)] = blockArray

                    x += blockWidth

                y += blockHeight

    return blocks


def applyDCT(blocks):

    for component in ["Y", "U", "V"]:

        for block in blocks[component]:
            blocks[component][block] = dct(dct(blocks[component][block], axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')

    return blocks


def quantization(blocks):

    quantizationMatrices = {}

    for component in ["Y", "U", "V"]:

        for key, block in blocks[component].items():

            blockSize = block.shape

            if blockSize not in quantizationMatrices:
                quantizationMatrices[blockSize] = quant.getQuantizationMatrix(blockSize, quant.DefaultQuantizationFunction)

            blocks[component][key] = np.round(block / quantizationMatrices[blockSize])

    return blocks




def inplaceCompress(block, stream, dcPred, dcEncoder, acEncoder):
    
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

    dcDelta = block[0] - dcPred.prediction
    dcPred.prediction = block[0]

    dcOffset = dcDelta < 0
    dcAbsDelta = abs(dcDelta)
    dcMagnitude = dcAbsDelta.item().bit_length()

    coeffBaseDifference = [0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767]
    entropyPositiveBase = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    dcEncoder.recordToken(dcMagnitude)
    writeByte(dcMagnitude, stream)

    cursor = 1

    if dcMagnitude != 0:
        
        dcOffset = dcDelta - coeffBaseDifference[dcMagnitude] if dcDelta < entropyPositiveBase[dcMagnitude] else dcDelta
        writeShort(dcOffset, stream[cursor:])

        cursor += 2
    
    hasEOB = False
    blockSize = len(block)

    eobIndex = blockSize - 1

    while eobIndex > 0:

        if block[eobIndex] != 0:
            break

        eobIndex -= 1

    if eobIndex == 0:
        hasEOB = True
        blockSize = 0
    elif eobIndex + 16 < blockSize:
        hasEOB = True
        blockSize = eobIndex + 1

    
    acIndex = 1

    while (acIndex < blockSize):
        
        ac = block[acIndex]
        acIndex += 1
        zeroes = 15
        sign = ac < 0
        
        if sign:
            ac = -ac

        #Count number of trailing zeroes
        for i in range(15):

            if acIndex >= blockSize:
                zeroes = i
                break

            if block[acIndex] != 0:
                zeroes = i
                break

            acIndex += 1

        #We have our zero count now, encode it
        if (abs(ac) <= 3):

            #No extended sequence
            seq = ((ac & 0x3) << 4) | zeroes

            if sign:
                seq |= 0x40

            acEncoder.recordToken(seq)
            writeByte(seq, stream[cursor:])
            cursor += 1

        else:

            #Extended sequence, split value
            if (ac > 1023):
                ac = 1023

            j0 = ac & 0xFF
            j1 = ac >> 8
            ext = 0x80 | (sign << 6) | (j1 << 4) | zeroes

            acEncoder.recordToken(ext)
            acEncoder.recordToken(j0)

            writeByte(ext, stream[cursor:])
            writeByte(j0, stream[cursor + 1:])
            cursor += 2

    if hasEOB:
        acEncoder.recordToken(0x4F)
        writeByte(0x4F, stream[cursor:])
        cursor += 1

    return cursor



        
def writeStreamHeader(header, bitBuffer: BitBuffer):

    '''
        The header has the following layout:
        char[3]: "SVC"
        u8:  Version
        u32: Luma Width
        u32: Luma Height
        u32: Chroma Width
        u32: Chroma Height
        u32: Frame Count
    '''
    bitBuffer.write(8, ord('S'))
    bitBuffer.write(8, ord('V'))
    bitBuffer.write(8, ord('C'))
    bitBuffer.write(8, 0x00)
    bitBuffer.write(8, header.gopSize)
    bitBuffer.write(32, header.lumaSize[0])
    bitBuffer.write(32, header.lumaSize[1])
    bitBuffer.write(32, header.chromaSize[0])
    bitBuffer.write(32, header.chromaSize[1])
    bitBuffer.write(32, header.frameCount)





def entropyCompress(iBlocks, header):

    # Estimate the maximum size that can ever be used for a single block
    compressBufferSize = (header.lumaPixels + header.chromaPixels * 2) * 4
    compressBuffer = np.empty(compressBufferSize, dtype = np.uint8)

    zigzagTransforms = {}
    zigzagBuffer = np.empty(64 * 64, dtype = np.int16)

    # Create bit buffer and write the stream header
    bitBuffer = BitBuffer()
    writeStreamHeader(header, bitBuffer)


    # Rearrange block tables to [component][frame][x, y] -> block data
    blockArrays = {}

    for component, blockTable in iBlocks.items():

        blockArrays[component] = {}

        for (frame, x, y), block in blockTable.items():

            if frame not in blockArrays[component]:
                blockArrays[component][frame] = {}

            blockArrays[component][frame][(x, y)] = block



    # Loop over all frames
    for frameID in range(header.frameCount):

        newFrame = True

        cursor = 0
        lumaEnd = 0

        biEncoder = HuffmanEncoder()
        dcEncoder = HuffmanEncoder()
        acEncoder = HuffmanEncoder()

        if frameID & (header.gopSize - 1):
            continue

        # Squeeze
        for component in ["Y", "U", "V"]:

            dcPrediction = DCPredictor()

            luma = component == "Y"
            baseBlockShift = 4 if luma else 3


            for blockPos, block in sorted(blockArrays[component][frameID / header.gopSize].items(), key = lambda x : x[0][1]):

                blockSize = (block.shape[1], block.shape[0])
                pixelCount = blockSize[0] * blockSize[1]

                # Write block information
                compressedBlockInfo = (blockSize[0] >> baseBlockShift).bit_length() | ((blockSize[1] >> baseBlockShift).bit_length() << 2) | (newFrame << 7)
                newFrame = False

                biEncoder.recordToken(compressedBlockInfo)
                writeByte(compressedBlockInfo, compressBuffer[cursor:])
                cursor += 1

                # If block transform indices haven't been calculated yet
                if blockSize not in zigzagTransforms:

                    # Add transform array to known zigzag sequences
                    zigzagTransforms[blockSize] = zz.zigzagTransform(blockSize)

                # Calculate zigzag transform
                zigzag = zigzagTransforms[blockSize]
                flatBlock = np.reshape(block, pixelCount)
                zigzagIndex = 0

                for i in range(pixelCount):
                    zigzagBuffer[i] = flatBlock[zigzag[i]]

                # Compress block
                cursorAdvance = inplaceCompress(zigzagBuffer[:pixelCount], compressBuffer[cursor:], dcPrediction, dcEncoder, acEncoder)
                cursor += cursorAdvance

            if component == "Y":
                lumaEnd = cursor


        # Build huffman trees
        biEncoder.buildTree()
        dcEncoder.buildTree()
        acEncoder.buildTree()

        # Write huffman tables
        bitBuffer.flush()
        huffmanStart = bitBuffer.size()

        biEncoder.serialize(bitBuffer)
        dcEncoder.serialize(bitBuffer)
        acEncoder.serialize(bitBuffer)

        bitBuffer.flush()
        huffmanEnd = bitBuffer.size()
        huffmanSize = huffmanEnd - huffmanStart

        # Setup statistics
        uncompressedSize = cursor
        compressedSize = 0

        # Compress
        compressCursor = 0
        baseBlockDim = 8

        while compressCursor < cursor:

            if compressCursor >= lumaEnd:
                baseBlockDim = 4

            sizeCompressed = compressBuffer[compressCursor]

            blockWidth = (baseBlockDim << (sizeCompressed & 0x3)).astype("int32")
            blockHeight = (baseBlockDim << ((sizeCompressed >> 2) & 0x3)).astype("int32")

            # Write frame header
            (frameCode, frameLength) = biEncoder.getCode(sizeCompressed)
            bitBuffer.write(frameLength, frameCode)

            compressedSize += frameLength
            compressCursor += 1

            # Write DC coefficient
            dcMagnitude = compressBuffer[compressCursor]
            (magCode, magLength) = dcEncoder.getCode(dcMagnitude)
            bitBuffer.write(magLength, magCode)

            compressedSize += magLength
            compressCursor += 1

            if dcMagnitude != 0:

                bitBuffer.write(dcMagnitude, readShort(compressBuffer[compressCursor:]))

                compressedSize += dcMagnitude
                compressCursor += 2

            # Huffman-compress AC coefficients
            acIndex = 1
            totalCoeffs = blockWidth * blockHeight

            while acIndex < totalCoeffs:

                v = compressBuffer[compressCursor]

                (code, length) = acEncoder.getCode(v)
                bitBuffer.write(length, code)

                compressedSize += length
                compressCursor += 1

                # Extended sequence
                if v & 0x80:

                    (extCode, extLength) = acEncoder.getCode(compressBuffer[compressCursor])
                    bitBuffer.write(extLength, extCode)

                    compressedSize += extLength
                    compressCursor += 1

                # EOB
                elif v == 0x4F:
                    break

                acIndex += (v & 0xF) + 1

        compressedSize = (compressedSize + 7) // 8
        compressedSize += huffmanSize
        uncompressedSize += huffmanSize

        print("Frame: {}, Full: {}, Squeezed: {}, Compressed: {}, Temporal Ratio: {}, Compression Ratio: {}".format(frameID, header.framePixels, uncompressedSize, compressedSize, uncompressedSize / compressedSize, header.framePixels / compressedSize))

    bitstream = bitBuffer.toBuffer()

    originalSize = header.totalPixels
    encodedSize = bitstream.size

    print("Compressed {} frames".format(header.frameCount))
    print("Full size: {}, Compressed: {}, Compression Ratio: {}".format(originalSize, encodedSize, originalSize / encodedSize))

    return bitstream