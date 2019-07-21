import random

class Link:
    
    CHANGE_STEP = 0.01
    LOST_STEP = 0.0001

    def __init__(self, name, feature, inputField, idx , pos, weight = -100):
        self.name = name
        self.feature = feature
        self.inputField = inputField
        self.idx = idx
        self.pos = pos
        if (weight == -100):
            self.weight = 0.5 + random.uniform(-0.2, 0.2)
        else:
            self.weight = weight

    def upWeight(self):
        self.weight += Link.CHANGE_STEP
        if self.weight > 1:
            self.weight = 1
    
    def downWeight(self):
        self.weight -= Link.CHANGE_STEP
    
    def lostWeight(self):
        self.weight -= Link.LOST_STEP