import numpy as np

class Field:

    def __init__(self, name, vision, vx, vy):
        self.name = name
        self.vision = vision
        self.vx = vx
        self.vy = vy
        self.visionSize = self.vision.fieldSize
        self.data = np.zeros((self.visionSize,self.visionSize)) 

    def run(self):
        x = self.vx*self.visionSize
        y = self.vy*self.visionSize
        self.data = self.vision.getData()[x:x+self.visionSize,y:y+self.visionSize]

    def getData(self):
        return self.data

    def getSize(self):
        return self.visionSize
