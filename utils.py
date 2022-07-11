class DCPredictor:
    
    def __init__(self):
        self.prediction = 0



def writeByte(x, stream):
    stream[0] = x

def writeShort(x, stream):
    stream[0] = x & 0xFF
    stream[1] = x >> 8

def writeInt(x, stream):
    stream[0] = x & 0xFF
    stream[1] = (x >> 8) & 0xFF
    stream[2] = (x >> 16) & 0xFF
    stream[3] = (x >> 24) & 0xFF

def readByte(stream):
    return stream[0]

def readShort(stream):
    return stream[0] | (stream[1] << 8)

def readInt(stream):
    return stream[0] | (stream[1] << 8) | (stream[2] << 16) | (stream[3] << 24)


def reverse(x, size):

    y = 0

    for i in range(size):
        y <<= 1
        y |= x & 0x1
        x >>= 1

    return y


def pBlockSize(ctuSize, luma):
    return ctuSize if luma else (ctuSize[0] >> 1, ctuSize[1] >> 1)