import sys
from tabnanny import check
import numpy as np
from bitbuffer import BitBuffer
from scipy.fftpack import dct
from header import Header
from huffman import HuffmanEncoder
from pblock import PBlock
import quantization as quant
from utils import *
import zigzag as zz
import cv2 as cv



def encoder(video):

    '''
        Encodes the video and returns a bitstream
    '''

    print("Encoding")

    header = Header((video["Y"].shape[2], video["Y"].shape[1]), (video["U"].shape[2], video["U"].shape[1]), video["Y"].shape[0], (16, 16), 8, 127, 0.90)

    (iFrames, pFrames) = splitFrames(video, header.gopSize)
    pBlocks = generatePBlocks(iFrames, pFrames, header)
    print("Generated P-Blocks")

    iBlocks = generateIBlocks(iFrames, header)
    print("Generated I-Blocks")

    bitstream = entropyCompress(iBlocks, pBlocks, header)
    print("Compressed")

    return bitstream



def splitFrames(video, gopSize):

    frameCount = video["Y"].shape[0]
    iFrameCount = (frameCount + (gopSize - 1)) // gopSize
    pFrameCount = frameCount - iFrameCount
    
    iFrames = {}
    pFrames = {}

    for component, frames in video.items():

        iFrames[component] = np.empty((iFrameCount, frames.shape[1], frames.shape[2]))
        pFrames[component] = np.empty((pFrameCount, frames.shape[1], frames.shape[2]))
        pIndex = 0

        for i in range(frameCount):
            
            if i & (gopSize - 1):
                pFrames[component][pIndex] = frames[i]
                pIndex += 1

            else:
                iFrames[component][i // gopSize] = frames[i]

    return iFrames, pFrames



def generatePBlocks(iFrames, pFrames, header):

    pBlocks = {"Y": {}, "U": {}, "V": {}}

    for component, frames in pFrames.items():

        luma = component == "Y"
        blockSize = pBlockSize(header.ctuSize, luma)
        frameSize = header.lumaSize if luma else header.chromaSize

        motionDistance = (header.pMotionMaxBlockOffset, header.pMotionMaxBlockOffset)

        for i in range(len(frames)):
            
            refIFrameIndex = i // (header.gopSize - 1)
            refIFrameID = refIFrameIndex * header.gopSize

            frameID = i + refIFrameIndex + 1
            refIFrame = iFrames[component][refIFrameIndex]

            pBlocks[component][frameID] = []

            for y in range(0, frameSize[1], blockSize[1]):

                for x in range(0, frameSize[0], blockSize[0]):

                    block = frames[i, y : y + blockSize[1], x : x + blockSize[0]].astype("float32")
                    regionBase = (max(x - motionDistance[0], 0), max(y - motionDistance[1], 0))
                    
                    if checkPerfectMatch(block, refIFrame[y : y + blockSize[1], x : x + blockSize[0]].astype("float32")):

                        # Do not code the block, perfect match
                        pBlocks[component][frameID].append(PBlock((x, y), 3, None, (0, 0)))

                    else:

                        (found, motion, diffBlock) = findMotionBlock(block, (x, y), refIFrame[regionBase[1] : min(y + motionDistance[1], frameSize[1]), regionBase[0] : min(x + motionDistance[0], frameSize[0])].astype("float32"), regionBase, header.pPatternThreshold)

                        if found:

                            diffBlock = np.round(dct(dct(diffBlock, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho') / quant.getQuantizationMatrix(diffBlock.shape))

                            if not np.any(diffBlock):

                                if motion != (0, 0):

                                    # Only code motion vector
                                    pBlocks[component][frameID].append(PBlock((x, y), 1, None, motion))

                                else:

                                    # Do not code the block
                                    pBlocks[component][frameID].append(PBlock((x, y), 3, None, (0, 0)))

                            else:

                                # Code motion p block
                                pBlocks[component][frameID].append(PBlock((x, y), 0, diffBlock, motion))

                        else:

                            # Code intra block
                            pBlocks[component][frameID].append(PBlock((x, y), 2, np.round(dct(dct(block, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho') / quant.getQuantizationMatrix(block.shape)), (0, 0)))

    return pBlocks



def checkPerfectMatch(pBlock, refRegion):
    return np.allclose(pBlock, refRegion)



def findMotionBlock(pBlock, pBlockPos, refRegion, refRegionPos, threshold):
    
    result = cv.matchTemplate(refRegion, pBlock, cv.TM_CCOEFF_NORMED)
    minValue, maxValue, minPos, maxPos = cv.minMaxLoc(result)

    if maxValue >= threshold:
        
        motionVector = (pBlockPos[0] - (refRegionPos[0] + maxPos[0]), pBlockPos[1] - (refRegionPos[1] + maxPos[1]))
        refBlock = refRegion[maxPos[1] : maxPos[1] + pBlock.shape[0], maxPos[0] : maxPos[0] + pBlock.shape[1]]

        return (True, motionVector, pBlock - refBlock)

    else:

        return (False, (0, 0), None)

    


def generateIBlocks(iFrames, header):

    blockData = divideIntraBlocks(header, iFrames)

    return applyDCTAndQuantize(blockData)




def divideIntraBlocks(header, iFrames):

    blockSize = header.ctuSize
    subsamplingFactor = (header.lumaSize[0] // header.chromaSize[1], header.lumaSize[1] // header.chromaSize[1])

    blockSizes = {
        "Y": blockSize,
        "U": (blockSize[0] // subsamplingFactor[0], blockSize[1] // subsamplingFactor[1]),
        "V": (blockSize[0] // subsamplingFactor[0], blockSize[1] // subsamplingFactor[1])
    }

    blocks = { "Y": {}, "U": {}, "V": {}}
    frameCount = iFrames["Y"].shape[0]

    for component, frames in iFrames.items():

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


def applyDCTAndQuantize(blocks):

    for component in ["Y", "U", "V"]:

        for block in blocks[component]:
            
            dctBlock = dct(dct(blocks[component][block], axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
            blocks[component][block] = np.round(dctBlock / quant.getQuantizationMatrix(dctBlock.shape))

    return blocks



def inplaceCompress(block, stream, dcPred, dcEncoder, acEncoder):
    
    '''
        The first element is the DC component, the rest is AC
        For DC, do a prediction-based scheme
        For AC, compress data as follows:
        ESXXZZZZ [XXXXXXXX]
        E: Extended sequence
        S: Sign bit
        X: Value
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
        u8: CTU Width
        u8: CTU Height
        u8: GoP Size
    '''
    bitBuffer.write(8, ord('S'))
    bitBuffer.write(8, ord('V'))
    bitBuffer.write(8, ord('C'))
    bitBuffer.write(8, 0x00)
    bitBuffer.write(32, header.lumaSize[0])
    bitBuffer.write(32, header.lumaSize[1])
    bitBuffer.write(32, header.chromaSize[0])
    bitBuffer.write(32, header.chromaSize[1])
    bitBuffer.write(32, header.frameCount)
    bitBuffer.write(8, header.ctuSize[0])
    bitBuffer.write(8, header.ctuSize[1])
    bitBuffer.write(8, header.gopSize)





def entropyCompress(iBlocks, pBlocks, header):

    # Estimate the maximum size that can ever be used for a single block
    compressBufferSize = (header.lumaPixels + header.chromaPixels * 2) * 4
    compressBuffer = np.empty(compressBufferSize, dtype = np.uint8)

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


    compressedSize = bitBuffer.size()

    # Loop over all frames
    for frameID in range(header.frameCount):

        cursor = 0
        compressCursor = 0

        lumaEnd = 0

        uncompressedSize = 0
        huffmanSize = 0

        if frameID & (header.gopSize - 1):
        
            # P-Frame coding
            mvEncoder = HuffmanEncoder()
            dcEncoder = HuffmanEncoder()
            acEncoder = HuffmanEncoder()

            for component in ["Y", "U", "V"]:

                dcPrediction = DCPredictor()
                pFrameArray = pBlocks[component][frameID]

                for pBlock in pFrameArray:

                    type = pBlock.type

                    compressBuffer[cursor] = type
                    cursor += 1

                    if type <= 1:

                        motionVector = pBlock.motionVector

                        compressBuffer[cursor] = motionVector[0]
                        compressBuffer[cursor + 1] = motionVector[1]

                        mvEncoder.recordToken(abs(motionVector[0]))
                        mvEncoder.recordToken(abs(motionVector[1]))

                        cursor += 2

                    if (type & 1) == 0:

                        block = pBlock.block
                        blockSize = (block.shape[1], block.shape[0])
                        pixelCount = blockSize[0] * blockSize[1]

                        zigzag = zz.zigzag(blockSize)
                        flatBlock = np.reshape(block, pixelCount)

                        for i in range(pixelCount):
                            zigzagBuffer[i] = flatBlock[zigzag[i]]

                        cursor += inplaceCompress(zigzagBuffer[:pixelCount], compressBuffer[cursor:], dcPrediction, dcEncoder, acEncoder)


                if component == "Y":
                    lumaEnd = cursor

            # Build huffman trees
            mvEncoder.buildTree()
            dcEncoder.buildTree()
            acEncoder.buildTree()

            # Write huffman tables
            bitBuffer.flush()
            huffmanStart = bitBuffer.size()

            mvEncoder.serialize(bitBuffer)
            dcEncoder.serialize(bitBuffer)
            acEncoder.serialize(bitBuffer)

            huffmanEnd = bitBuffer.size()
            huffmanSize = huffmanEnd - huffmanStart

            uncompressedSize = huffmanSize + cursor

            blockSize = pBlockSize(header.ctuSize, True)
            pixelCount = blockSize[0] * blockSize[1]

            # Compress
            while compressCursor < cursor:

                if compressCursor >= lumaEnd:
                    blockSize = pBlockSize(header.ctuSize, False)
                    pixelCount = blockSize[0] * blockSize[1]

                type = compressBuffer[compressCursor]
                compressCursor += 1
                
                bitBuffer.write(2, type)

                if type <= 1:

                    # Write motion vector
                    for i in range(2):

                        motionDistance = compressBuffer[compressCursor].astype("int32")

                        if motionDistance >= 128:
                            motionDistance -= 256

                        bitBuffer.write(1, motionDistance >= 0)

                        (mvCode, mvLength) = mvEncoder.getCode(abs(motionDistance))
                        bitBuffer.write(mvLength, mvCode)

                        compressCursor += 1

                if (type & 1) == 0:
                    compressCursor += huffmanCompressBlock(dcEncoder, acEncoder, bitBuffer, compressBuffer[compressCursor:], pixelCount)

        else:

            # I-Frame coding
            biEncoder = HuffmanEncoder()
            dcEncoder = HuffmanEncoder()
            acEncoder = HuffmanEncoder()

            # Squeeze
            for component in ["Y", "U", "V"]:

                dcPrediction = DCPredictor()

                luma = component == "Y"
                baseBlockShift = 4 if luma else 3


                for blockPos, block in sorted(blockArrays[component][frameID // header.gopSize].items(), key = lambda x : x[0][1]):

                    blockSize = (block.shape[1], block.shape[0])
                    pixelCount = blockSize[0] * blockSize[1]

                    # Write block information
                    compressedBlockInfo = (blockSize[0] >> baseBlockShift).bit_length() | ((blockSize[1] >> baseBlockShift).bit_length() << 2)
                    newFrame = False

                    biEncoder.recordToken(compressedBlockInfo)
                    writeByte(compressedBlockInfo, compressBuffer[cursor:])
                    cursor += 1

                    # Calculate zigzag transform
                    zigzag = zz.zigzag(blockSize)
                    flatBlock = np.reshape(block, pixelCount)
                    zigzagIndex = 0

                    for i in range(pixelCount):
                        zigzagBuffer[i] = flatBlock[zigzag[i]]

                    # Compress block
                    cursor += inplaceCompress(zigzagBuffer[:pixelCount], compressBuffer[cursor:], dcPrediction, dcEncoder, acEncoder)

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

            huffmanEnd = bitBuffer.size()
            huffmanSize = huffmanEnd - huffmanStart

            uncompressedSize = huffmanSize + cursor
            baseBlockDim = 8

            # Compress
            while compressCursor < cursor:

                if compressCursor >= lumaEnd:
                    baseBlockDim = 4

                sizeCompressed = compressBuffer[compressCursor]

                blockWidth = (baseBlockDim << (sizeCompressed & 0x3)).astype("int32")
                blockHeight = (baseBlockDim << ((sizeCompressed >> 2) & 0x3)).astype("int32")

                # Write frame header
                (frameCode, frameLength) = biEncoder.getCode(sizeCompressed)
                bitBuffer.write(frameLength, frameCode)

                compressCursor += 1
                compressCursor += huffmanCompressBlock(dcEncoder, acEncoder, bitBuffer, compressBuffer[compressCursor:], blockWidth * blockHeight)

        frameCompressedSize = bitBuffer.size() - compressedSize
        compressedSize += frameCompressedSize

        print("Frame: {}, Full: {}, Squeezed: {}, Compressed: {}, Huffman Ratio: {}, Compression Ratio: {}".format(frameID, header.framePixels, uncompressedSize, frameCompressedSize, uncompressedSize / frameCompressedSize, header.framePixels / frameCompressedSize))

    bitstream = bitBuffer.toBuffer()

    originalSize = header.totalPixels
    encodedSize = bitstream.size

    print("Compressed {} frames".format(header.frameCount))
    print("Full size: {}, Compressed: {}, Compression Ratio: {}".format(originalSize, encodedSize, originalSize / encodedSize))

    return bitstream




def huffmanCompressBlock(dcEncoder, acEncoder, bitBuffer, compressBuffer, totalCoeffs):

    
    # Write DC coefficient
    dcMagnitude = compressBuffer[0]
    (magCode, magLength) = dcEncoder.getCode(dcMagnitude)
    bitBuffer.write(magLength, magCode)

    cursor = 1

    if dcMagnitude != 0:

        bitBuffer.write(dcMagnitude, readShort(compressBuffer[cursor:]))

        cursor += 2


    # Huffman-compress AC coefficients
    acIndex = 1

    while acIndex < totalCoeffs:

        v = compressBuffer[cursor]

        (code, length) = acEncoder.getCode(v)
        bitBuffer.write(length, code)

        cursor += 1

        # Extended sequence
        if v & 0x80:

            (extCode, extLength) = acEncoder.getCode(compressBuffer[cursor])
            bitBuffer.write(extLength, extCode)

            cursor += 1

        # EOB
        elif v == 0x4F:
            break

        acIndex += (v & 0xF) + 1

    return cursor