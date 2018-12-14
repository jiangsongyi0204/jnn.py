import numpy as np

class Field:

    def __init__(self, name, vision, vx, vy):
        self.name = name
        self.vision = vision
        self.vx = vx
        self.vy = vy
        self.visionSize = self.vision.size
        self.data = np.zeros((self.visionSize,self.visionSize)) 
    
    def getData(self):
        for x in range(0,self.visionSize):
            for y in range (0,self.visionSize):
                self.data[x][y] = self.vision.getData()[x+self.vx*self.visionSize,y+self.vy*self.visionSize]
        return self.data

    def getSize(self):
        return self.visionSize
