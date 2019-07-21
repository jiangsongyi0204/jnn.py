import random
import cv2
import numpy as np

class Feature:

    def __init__(self, name, sensor, pattern):
        self.name = name
        self.sensor = sensor
        self.data = np.zeros((100,100))
        self.output = np.zeros((100,100))
        self.pattern = np.array(pattern) #np.array([[0,0,1],[0,1,0],[1,0,0]])
        self.patternSize = self.pattern.shape[0]
        self.patternSize2 = int(self.patternSize/2)
        self.patternCount = self.pattern.sum()

    def match(self):
        self.data = cv2.Canny(self.sensor.data.astype(np.uint8), 100, 200)
        for x in range(self.patternSize2,100-self.patternSize2):
            for y in range(self.patternSize2,100-self.patternSize2):
                self.output[x,y] = (((self.data[x-self.patternSize2:x+self.patternSize2+1,y-self.patternSize2:y+self.patternSize2+1] & self.pattern) >= 1).sum() == self.patternCount).astype(np.float32)

        
