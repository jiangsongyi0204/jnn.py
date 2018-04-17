import cv2
import numpy as np

class Sensor:

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.imgSize = int(np.sqrt(self.size))
        self.inputData = []
        if self.imgSize * self.imgSize < self.size:
            raise("Sensor size error!")
    
    def readImage(self, image):
        resize_image = cv2.resize(image,(self.imgSize, self.imgSize))
        edged_image = cv2.Canny(resize_image,100,150)
        self.inputData = []
        for x in range(self.imgSize):
            for y in range(self.imgSize):
                if edged_image[x,y] > 0:
                    self.inputData.append(1.0)
                else:
                    self.inputData.append(0.0)

    def getSensorImg(self):
        return np.reshape(self.inputData, (self.imgSize, self.imgSize))

    def debug(self):
        print(self.name + ":" + ''.join(self.inputData))