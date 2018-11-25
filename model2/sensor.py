import numpy as np

class Sensor:

    def __init__(self, name):
        self.name = name
        self.data = []
    
    def read(self, image):
        if len(self.data) > 10:
            self.data.pop(0)
        self.data.append(image)

    def getShape(self):
        return np.array(self.data[0]).shape
           
    def getOutput(self):
        return np.array(self.data[-1])
