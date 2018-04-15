import random

class Link:
    def __init__(self, name, sensor, pos, featuremcell):
        self.name = name
        self.sensor = sensor
        self.pos = pos
        self.featuremcell = featuremcell
        self.weight = 0.5 + random.uniform(-0.2, 0.2)

    def upWeight(self):
        self.weight += 0.01
    
    def downWeight(self):
        self.weight -= 0.01