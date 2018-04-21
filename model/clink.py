import random

class CellLink:
    
    CHANGE_STEP = 0.01

    def __init__(self, name, from_fcc, to_fcc):
        self.name = name
        self.from_fcc = from_fcc
        self.to_fcc = to_fcc
        self.weight = 0.5 + random.uniform(-0.2, 0.2)

    def upWeight(self):
        self.weight += CellLink.CHANGE_STEP
        if self.weight > 1:
            self.weight = 1
    
    def downWeight(self):
        self.weight -= CellLink.CHANGE_STEP