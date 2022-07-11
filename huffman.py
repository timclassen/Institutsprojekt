from configparser import ExtendedInterpolation
import numpy as np

from bitbuffer import BitBuffer, BitReader
from utils import reverse



# Canonical Huffman Encoder
class HuffmanEncoder:

    def __init__(self):
        self.probabilityTable = {}
        self.symbolTable = {}


    def recordToken(self, token):
        
        if token not in self.probabilityTable:
            self.probabilityTable[token] = 1
        else:
            self.probabilityTable[token] += 1


    def buildTree(self):

        if len(self.probabilityTable.items()) == 0:
            return

        symbolList = sorted((n, symbol) for symbol, n in self.probabilityTable.items())

        while len(symbolList) > 1:

            left = symbolList[0]
            right = symbolList[1]
            newKey = left[0] + right[0]
            symbolList = symbolList[2:]

            insertionIndex = 0

            for (n, symbol) in symbolList:

                if n < newKey:
                    insertionIndex += 1

            symbolList.insert(insertionIndex, (newKey, (left[1], right[1])))

        self.symbolTable = {}

        codeLength = 1

        symbolRoot = symbolList[0][1]

        if isinstance(symbolRoot, tuple):

            treeTable = []
            self.convertTree(1, symbolRoot, treeTable)
            treeTable.sort()

            code = 0
            prevLength = 1

            for (length, symbol) in treeTable:

                if length != prevLength:
                    code <<= length - prevLength
                    prevLength = length

                self.symbolTable[symbol] = (reverse(code, length), length)
                code += 1

        else:

            self.symbolTable[symbolRoot] = (0, 1)


    def convertTree(self, length, symlist, treeTable):

        left = symlist[0]
        right = symlist[1]

        if isinstance(left, tuple):
            self.convertTree(length + 1, left, treeTable)
        else:
            treeTable.append((length, left))

        if isinstance(right, tuple):
            self.convertTree(length + 1, right, treeTable)
        else:
            treeTable.append((length, right))



    def getCode(self, symbol):
        return self.symbolTable[symbol]


    def serialize(self, bitBuffer: BitBuffer):

        if len(self.symbolTable) == 0:
            bitBuffer.write(8, 0)
            return
            
        highestLength = next(reversed(self.symbolTable.values()))[1]
        lengths = np.zeros(highestLength, dtype = np.uint8)

        for symbol, (code, length) in self.symbolTable.items():
            lengths[length - 1] += 1

        bitBuffer.write(8, highestLength)
        
        for i in range(lengths.size):
            bitBuffer.write(8, lengths[i])

        for symbol, (code, length) in self.symbolTable.items():
            bitBuffer.write(8, symbol)
            




# Fast Huffman Decoder
class HuffmanDecoder:

    
    def __init__(self):
        self.fastHuffmanTable = np.full(256, (0, 0xFF), dtype = (np.uint8, 2))
        self.extHuffmanTables = []
        self.maxExtLength = 0


    def deserialize(self, bitReader: BitReader):

        highestLength = bitReader.read(8)

        lengths = []

        for i in range(highestLength):
            lengths.append(bitReader.read(8))

        code = 0
        extGenerated = False
        self.maxExtLength = highestLength - 8

        '''
            Fast huffman tables, algorithm based off Arclight's JPEGDecoder
            The idea is breaking up a tree that has a O(log n) huffman code retrieval cost into a O(1) operation.
            We cut the tree after 8 levels and fill the fast table with all the codes derived from a given
            8-bit value. Each entry consists of a tuple (symbol, length).
        '''

        for length in range(highestLength):

            code <<= 1
            count = lengths[length]
            realLength = length + 1

            for i in range(count):

                symbol = bitReader.read(8)

                revCode = reverse(code, realLength)

                if length < 8:

                    endCode = 1 << (7 - length)

                    for j in range(endCode):
                        self.fastHuffmanTable[(j << realLength) + revCode] = (symbol, realLength)

                else:

                    if not extGenerated:

                        for j in range(256):

                            if self.fastHuffmanTable[j][1] == 0xFF:
                                
                                extIndex = len(self.extHuffmanTables)
                                self.fastHuffmanTable[j] = (extIndex, 0xFF)
                                self.extHuffmanTables.append(np.empty(2 << self.maxExtLength, dtype = (np.uint8, 2)))

                        extGenerated = True

                    fastPrefix = revCode & 0xFF
                    extCode = revCode >> 8

                    extLength = length - 7
                    endCode = 1 << (self.maxExtLength - extLength)

                    extTableID = self.fastHuffmanTable[fastPrefix][0]

                    for j in range(endCode):
                        self.extHuffmanTables[extTableID][(j << extLength) + extCode] = (symbol, realLength)

                code += 1


    def read(self, bitReader: BitReader):

        result = bitReader.peek(8)
        (symbol, length) = self.fastHuffmanTable[result]

        if length == 0xFF:

            bitReader.consume(8)
            result = bitReader.peek(self.maxExtLength)
            (symbol, length) = self.extHuffmanTables[symbol][result]

            bitReader.consume(length - 8)

        else:

            bitReader.consume(length)

        return symbol
