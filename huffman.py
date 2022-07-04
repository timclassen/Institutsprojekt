#Implements huffman coding

class HuffmanEncoder:

    probabilityTable = {}
    symbolTable = {}

    def clearTokens(self):
        self.probabilityTable.clear()


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

        code = 0
        codeLength = 1

        self.calculateCode(symbolList[0][1], code, codeLength)

        self.symbolTable = dict(sorted(self.symbolTable.items(), key = lambda v: v[1][1]))

        #for symbol, (code, length) in self.symbolTable:
        #    print("{}: L = {}, {:b}".format(symbol, length, code))


    def calculateCode(self, symlist, code, length):

        left = symlist[0]
        right = symlist[1]

        if left == 80:
            True

        if isinstance(left, tuple):
            self.calculateCode(left, code << 1, length + 1)
        else:
            self.symbolTable[left] = (code << 1, length)

        if isinstance(right, tuple):
            self.calculateCode(right, (code << 1) + 1, length + 1)
        else:
            self.symbolTable[right] = ((code << 1) + 1, length)


    def getCode(self, symbol):
        return self.symbolTable[symbol]