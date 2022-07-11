import numpy as np



# Bit Buffer
class BitBuffer:

    def __init__(self):

        self.data = []
        self.buffer = 0
        self.offset = 0


    def size(self):
        return len(self.data) + (self.offset + 7) // 8


    def flush(self):

        while self.offset > 0:
            self.data.append(self.buffer & 0xFF)
            self.buffer = 0
            self.offset = 0

    def write(self, length, bits):
        
        self.buffer |= bits << self.offset
        self.offset += length

        for i in range(self.offset // 8):
            self.data.append(self.buffer & 0xFF)
            self.buffer >>= 8
            self.offset -= 8


    def toBuffer(self):
        
        self.flush()

        array = np.asarray(self.data, dtype = np.uint8)
        self.data.clear()

        return array



# Bit Reader
class BitReader:

    def __init__(self, data):
        self.data = data
        self.cursor = 0
        self.buffer = 0
        self.size = 0


    def align(self):

        shift = self.size & 0x7
        
        self.buffer >>= shift
        self.size -= shift


    def peek(self, length):

        while self.size < length and self.cursor < self.data.size:

            self.buffer |= self.data[self.cursor] << self.size
            self.cursor += 1
            self.size += 8

        return self.buffer & ((1 << length) - 1)
        

    def consume(self, length):
        self.buffer >>= length
        self.size -= length


    def read(self, length):

        result = self.peek(length)
        self.consume(length)

        return result
