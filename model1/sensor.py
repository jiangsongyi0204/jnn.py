import numpy as np

class Sensor:

    def __init__(self, name, showHistory=False):
        self.name = name
        self.showHistory = showHistory
        self.data = []
    
    def read(self, image):
        if self.showHistory == False:
            self.data = []
        else:
            if len(self.data)>30:
                self.data.pop(0)
        inputData = []
        for x in range(len(image)):
            for y in range(len(image[x])):
                if image[x,y] > 0:
                    inputData.append(1.0)
                else:
                    inputData.append(0.0)
        self.data.append(inputData)

    def getData(self):
        return self.data

    def getLength(self):
        return len(self.data)
    
    def getSize(self):
        return len(self.data[0])
           
    def getSensorImg(self):
        ret = []
        imgSize = int(np.sqrt(self.getSize()))
        for data in self.data:
            img = np.reshape(data, (imgSize, imgSize))
            if len(ret)==0:
                ret = img
            else:
                ret = np.concatenate((ret, img), axis=1)
        return ret
