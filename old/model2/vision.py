import numpy as np
import cv2
from model2.column import Column
from model2.field import Field

class Vision:

    def __init__(self, name, sensor):
        self.name = name
        self.sensor = sensor
        self.sensor_x = 0
        self.sensor_y = 0
        self.window = 0
        self.fields = []
        self.columns = []
        self.fieldSize = 40
        self.fieldLen = 10
        self.data = np.zeros((self.fieldSize*self.fieldLen,self.fieldSize*self.fieldLen))
        self.init()
    
    def init(self):
        for x in range(self.fieldLen):
            for y in range(self.fieldLen):
                field = Field(str(x)+"-"+str(y),self,x,y)
                self.fields.append(field)
                column = Column(str(x)+"-"+str(y),self,x,y,field)
                self.columns.append(column)

    def getData(self):
        return self.data

    def run(self):
        self.work()
        for column in self.columns:
            column.run()
        for column in self.columns:
            column.predict()            
        for column in self.columns:
            column.learn()

    def getColumnImg(self,tp="Feature"):      
        ret = np.zeros((self.fieldSize*self.fieldLen,self.fieldSize*self.fieldLen))
        for x in range(self.fieldLen):
            for y in range(self.fieldLen):
                if (tp=="Feature"): 
                    ret[x*self.fieldSize:x*self.fieldSize+self.fieldSize,y*self.fieldSize:y*self.fieldSize+self.fieldSize] = self.columns[x*self.fieldLen+y].getImg()
                elif (tp=="MatchedFeature"):
                    ret[x*self.fieldSize:x*self.fieldSize+self.fieldSize,y*self.fieldSize:y*self.fieldSize+self.fieldSize] = self.columns[x*self.fieldLen+y].getMatchedImg()
                elif (tp=="PredictedFeature"):
                    ret[x*self.fieldSize:x*self.fieldSize+self.fieldSize,y*self.fieldSize:y*self.fieldSize+self.fieldSize] = self.columns[x*self.fieldLen+y].getPredictedImg()
                else:
                    ret[x*self.fieldSize:x*self.fieldSize+self.fieldSize,y*self.fieldSize:y*self.fieldSize+self.fieldSize] = self.columns[x*self.fieldLen+y].getEdgeImg()
        retColor = np.full((self.fieldSize*self.fieldLen,self.fieldSize*self.fieldLen,3),[0.0,0.0,0.0])
        retColor[ret == 1] = [0.0  ,0.0  ,255.0]
        retColor[ret == 10] = [0.0  ,255.0  ,0.0]
        retColor[ret == 11] = [50.0  ,255.0  ,0.0]
        retColor[ret == 255] = [255.0  ,0.0  ,0.0]
        return retColor

    def work(self):
        sensordata = self.sensor.getOutput()
        self.data = cv2.resize(sensordata, dsize=(self.fieldSize*self.fieldLen,self.fieldSize*self.fieldLen), interpolation=cv2.INTER_LINEAR)
        #self.data = cv2.Canny(image.astype(np.uint8), 100, 200)

    def getColumnByPos(self,x,y):
        if (x < 0 or x >= self.fieldLen or y < 0 or y >= self.fieldLen):
            return None
        return self.columns[x*self.fieldLen+y]

    def getImg(self):
        return self.data