# TODO: Put your implementation of the encoder here
from distutils.ccompiler import gen_lib_options
from email.mime import base
import numpy as np
from scipy.fftpack import idct
from bitbuffer import BitReader
from huffman import HuffmanDecoder
from pblock import PBlock
import quantization as quant
from utils import DCPredictor, pBlockSize
import zigzag as zz
from header import Header



def decoder(bitstream):

    print("Decoding")

    (header, iBlocks, pBlocks) = entropyDecompress(bitstream)

    video = {
        "Y": np.zeros((header.frameCount, header.lumaSize[1], header.lumaSize[0]), dtype="uint8"),
        "U": np.zeros((header.frameCount, header.chromaSize[1], header.chromaSize[0]), dtype="uint8"),
        "V": np.zeros((header.frameCount, header.chromaSize[1], header.chromaSize[0]), dtype="uint8")
    }

    generateIFrames(video, iBlocks, header)
    print("Generated I-Frames")

    generatePFrames(video, pBlocks, header)
    print("Generated P-Frames")

    return video



def generateIFrames(video, iBlocks, header):

    dequantizeAndApplyIDCT(iBlocks)
    return reassembleIntraBlocks(video, header, iBlocks)



def reassembleIntraBlocks(video, header, iBlocks):

    for component in iBlocks:

        luma = component == "Y"
        frameSize = header.lumaSize if luma else header.chromaSize

        for block, blockData in iBlocks[component].items():

            frame = block[0]
            x = block[1]
            y = block[2]

            video[component][frame, y:y+blockData.shape[0], x:x+blockData.shape[1]] = np.clip(np.round(blockData[:min(blockData.shape[0], frameSize[1] - y), :min(blockData.shape[1], frameSize[0] - x)]), 0, 255)



def dequantizeAndApplyIDCT(blocks):

    for component in ["Y", "U", "V"]:
        for key, block in blocks[component].items():
            blocks[component][key] = idct(idct(block * quant.getQuantizationMatrix(block.shape), axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')



def generatePFrames(video, pBlocks, header):

    for component, frames in pBlocks.items():

        luma = component == "Y"
        blockSize = pBlockSize(header.ctuSize, luma)
        frameSize = header.lumaSize if luma else header.chromaSize

        for frameID in range(header.frameCount):

            if (frameID & (header.gopSize - 1)) == 0:
                continue

            refFrameID = frameID // header.gopSize * header.gopSize

            for pBlock in frames[frameID]:

                pos = pBlock.position
                block = pBlock.block

                if pBlock.type <= 1:
                    refPos = pos - pBlock.motionVector

                if (pBlock.type & 1) == 0:

                    block = idct(idct(block * quant.getQuantizationMatrix(block.shape), axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')

                    if pBlock.type == 0:
                        block += video[component][refFrameID, refPos[1] : refPos[1] + blockSize[1], refPos[0] : refPos[0] + blockSize[0]]

                    video[component][frameID, pos[1] : pos[1] + blockSize[1], pos[0] : pos[0] + blockSize[0]] = np.clip(np.round(block), 0, 255)

                else:

                    if pBlock.type == 1:
                        video[component][frameID, pos[1] : pos[1] + blockSize[1], pos[0] : pos[0] + blockSize[0]] = video[component][refFrameID, refPos[1] : refPos[1] + blockSize[1], refPos[0] : refPos[0] + blockSize[0]]
                    else:
                        video[component][frameID, pos[1] : pos[1] + blockSize[1], pos[0] : pos[0] + blockSize[0]] = video[component][refFrameID, pos[1] : pos[1] + blockSize[1], pos[0] : pos[0] + blockSize[0]]




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

    lumaSizeW = bitReader.read(32)
    lumaSizeH = bitReader.read(32)
    chromaSizeW = bitReader.read(32)
    chromaSizeH = bitReader.read(32)
    frameCount = bitReader.read(32)
    ctuSizeX = bitReader.read(8)
    ctuSizeY = bitReader.read(8)
    gopSize = bitReader.read(8)

    header = Header((lumaSizeW, lumaSizeH), (chromaSizeW, chromaSizeH), frameCount, (ctuSizeX, ctuSizeY), gopSize)

    return header



def entropyDecompress(bitstream):

    bitReader = BitReader(bitstream)
    header = readStreamHeader(bitReader)

    iBlocks = { "Y": {}, "U": {}, "V": {}}
    pBlocks = { "Y": {}, "U": {}, "V": {}}

    # For simplicity reasons, we assume blocks are in descending order towards the border
    blockLumaEndX = ((header.lumaSize[0] + 7) // 8) * 8
    blockLumaEndY = ((header.lumaSize[1] + 7) // 8) * 8
    blockChromaEndX = ((header.chromaSize[0] + 3) // 4) * 4
    blockChromaEndY = ((header.chromaSize[1] + 3) // 4) * 4
    
    for frameID in range(header.frameCount):

        interFrame = frameID & (header.gopSize - 1)

        bmDecoder = HuffmanDecoder()
        dcDecoder = HuffmanDecoder()
        acDecoder = HuffmanDecoder()

        bitReader.align()

        bmDecoder.deserialize(bitReader)
        dcDecoder.deserialize(bitReader)
        acDecoder.deserialize(bitReader)

        for component in ["Y", "U", "V"]:

            luma = component == "Y"
            baseBlockDim = 8 if luma else 4
            blockSize = pBlockSize(header.ctuSize, luma)

            dcPrediction = DCPredictor()
            pBlocks[component][frameID] = []

            blockEndX = blockLumaEndX if luma else blockChromaEndX
            blockEndY = blockLumaEndY if luma else blockChromaEndY
            blockPosition = (0, 0)

            while True:

                if interFrame:

                    blockWidth = blockSize[0]
                    blockHeight = blockSize[1]

                    type = bitReader.read(2)
                    pBlock = PBlock(blockPosition, type)

                    if type <= 1:

                        motionVector = np.zeros(2, dtype = np.int32)

                        for i in range(2):

                            positive = bitReader.read(1)
                            magnitude = bmDecoder.read(bitReader).astype("int32")

                            if not positive:
                                magnitude = -magnitude

                            motionVector[i] = magnitude

                        pBlock.motionVector = motionVector

                    if(type & 1) == 0:

                        block = np.zeros(blockWidth * blockHeight, dtype = np.float64)
                        inplaceDecompress(block, bitReader, dcPrediction, dcDecoder, acDecoder, zz.zigzag((blockWidth, blockHeight)))

                        pBlock.block = np.reshape(block, (blockHeight, blockWidth))

                    pBlocks[component][frameID].append(pBlock)

                else:

                    # Read block information
                    sizeCompressed = bmDecoder.read(bitReader)

                    blockWidth = (baseBlockDim << (sizeCompressed & 0x3)).astype("int32")
                    blockHeight = (baseBlockDim << ((sizeCompressed >> 2) & 0x3)).astype("int32")

                    # Setup block information
                    blockKey = (frameID, blockPosition[0], blockPosition[1])
                    blockSize = (blockWidth, blockHeight)

                    totalPixels = blockWidth * blockHeight

                    # Create block buffer
                    block = np.zeros(totalPixels, dtype = np.float64)

                    # Decompress
                    inplaceDecompress(block, bitReader, dcPrediction, dcDecoder, acDecoder, zz.zigzag(blockSize))

                    # Add block to tree
                    iBlocks[component][blockKey] = np.reshape(block, (blockHeight, blockWidth))

                # Set new block position
                blockPosition = (blockPosition[0] + blockWidth, blockPosition[1])

                # Jump to next block line
                if blockPosition[0] >= blockEndX:
                    blockPosition = (0, blockPosition[1] + blockHeight)

                if blockPosition[1] >= blockEndY:
                    break

        print("Decoded frame {}".format(frameID))

    return (header, iBlocks, pBlocks)