import numpy as np

class Sensor:

    def __init__(self, name):
        self.name = name
        self.data = None
    
    def read(self, image):
        self.data = image

    def getShape(self):
        return self.data.shape
           
    def getOutput(self):
        return self.data
