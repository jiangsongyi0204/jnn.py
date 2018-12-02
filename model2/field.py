import numpy as np

class Field:

    def __init__(self, name, vision, vx, vy):
        self.name = name
        self.vision = vision
        self.vx = vx
        self.vy = vy
        self.data = np.zeros((self.vision.r,self.vision.r)) 
    
    def getData(self):
        for x in range(0,self.vision.r):
            for y in range (0,self.vision.r):
                self.data[x][y] = self.vision.getData()[x+self.vx,y+self.vy]
        return self.data

    def getLength(self):
        return self.vision.r
    
    def getSize(self):
        return self.vision.r

    def getOutput(self):
        return self.getData()