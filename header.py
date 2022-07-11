#The video's metadata
class Header:

    def __init__(self, lumaSize, chromaSize, frames, ctuSize, gopSize, pFrameMaxBacktrack = 4, pMotionMaxBlockOffset = 16):
        self.lumaSize = lumaSize
        self.chromaSize = chromaSize
        self.frameCount = frames
        self.lumaPixels = lumaSize[0] * lumaSize[1]
        self.chromaPixels = chromaSize[0] * chromaSize[1]
        self.framePixels = self.lumaPixels + 2 * self.chromaPixels
        self.totalPixels = self.framePixels * self.frameCount
        self.ctuSize = ctuSize
        self.gopSize = gopSize
        self.pFrameMaxBacktrack = pFrameMaxBacktrack
        self.pMotionMaxBlockOffset = pMotionMaxBlockOffset