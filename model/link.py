import random

class Link:
    
    CHANGE_STEP = 0.01

    def __init__(self, name, sensor, pos, featuremcell):
        self.name = name
        self.sensor = sensor
        self.pos = pos
        self.featuremcell = featuremcell
        self.weight = 0.5 + random.uniform(-0.2, 0.2)

    def upWeight(self):
        self.weight += Link.CHANGE_STEP
        if self.weight > 1:
            self.weight = 1
    
    def downWeight(self):
        self.weight -= Link.CHANGE_STEP