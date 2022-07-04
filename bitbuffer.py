import numpy as np



class BitBuffer:

    def __init__(self):
        self.data = []
        self.buffer = 0
        self.offset = 0


    def write(self, length, bits):
        
        self.buffer |= bits << self.offset
        self.offset += length

        for i in range(self.offset // 8):
            self.data.append(self.buffer & 0xFF)
            self.buffer >>= 8
            self.offset -= 8

    def getBuffer(self):
        
        if self.offset > 0:
            self.data.append(self.buffer)
            self.buffer = 0
            self.offset = 0

        array = np.asarray(self.data, dtype = np.uint8)
        self.data.clear()

        return array