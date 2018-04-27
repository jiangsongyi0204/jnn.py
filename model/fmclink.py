import random

class FMCLink:
    
    CHANGE_STEP = 0.01

    def __init__(self, name, from_fmc, to_fmc):
        self.name = name
        self.from_fmc = from_fmc
        self.to_fmc = to_fmc
        self.weight = 0.5 + random.uniform(-0.2, 0.2)

    def upWeight(self):
        self.weight += FMCLink.CHANGE_STEP
        if self.weight > 1:
            self.weight = 1
    
    def downWeight(self):
        self.weight -= FMCLink.CHANGE_STEP