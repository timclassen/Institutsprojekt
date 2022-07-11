class PBlock:

    '''
        Type is one of the following:
        0: Motion-based differential coded P-Block
        1: Motion-based empty P-Block
        2: Intra-coded P-Block
        3: Empty P-Block
    '''

    def __init__(self, position = (0, 0), type = 0, block = None, motionVector = (0, 0)):
        self.position = position
        self.type = type
        self.block = block
        self.motionVector = motionVector
