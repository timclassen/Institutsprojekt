import numpy as np

from bitbuffer import BitBuffer



# Canonical Huffman coder
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

                self.symbolTable[symbol] = (code, length)
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

        highestLength = next(reversed(self.symbolTable.values()))[1]
        lengths = np.zeros(highestLength, dtype = np.uint8)

        for symbol, (code, length) in self.symbolTable.items():
            lengths[length - 1] += 1

        bitBuffer.write(8, highestLength)
        
        for i in range(lengths.size):
            bitBuffer.write(8, lengths[i])

        for symbol, (code, length) in self.symbolTable.items():
            bitBuffer.write(8, symbol)
            





class HuffmanDecoder:

    pass