#The video's metadata
class Header:

    def __init__(self, lumaSize, chromaSize, frames):
        self.lumaSize = lumaSize
        self.chromaSize = chromaSize
        self.frameCount = frames
        self.lumaPixels = lumaSize[0] * lumaSize[1]
        self.chromaPixels = chromaSize[0] * chromaSize[1]
        self.framePixels = self.lumaPixels + 2 * self.chromaPixels
        self.totalPixels = self.framePixels * self.frameCount