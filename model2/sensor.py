import numpy as np

class Sensor:

    def __init__(self, name):
        self.name = name
        self.data_old = None
        self.data = None
    
    def read(self, image):
        self.data_old = self.data
        self.data = image

    def getShape(self):
        return self.data.shape
           
    def getOutput(self):
        if self.data_old > 4:
            diff = np.array(self.data[-1]) - np.array(self.data[0])
            print(diff)
            return diff
        else:
            return np.zeros(self.getShape())
