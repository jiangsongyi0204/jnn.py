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
        self.fieldSize = 50
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

    def getColumnImg(self,tp=1):      
        ret = np.zeros((self.fieldSize*self.fieldLen,self.fieldSize*self.fieldLen))
        for x in range(self.fieldLen):
            for y in range(self.fieldLen):
                if (tp==1): 
                    ret[x*self.fieldSize:x*self.fieldSize+self.fieldSize,y*self.fieldSize:y*self.fieldSize+self.fieldSize] = self.columns[x*self.fieldLen+y].getImg()
                elif (tp==2):
                    ret[x*self.fieldSize:x*self.fieldSize+self.fieldSize,y*self.fieldSize:y*self.fieldSize+self.fieldSize] = self.columns[x*self.fieldLen+y].getMatchedImg()
                else:
                    ret[x*self.fieldSize:x*self.fieldSize+self.fieldSize,y*self.fieldSize:y*self.fieldSize+self.fieldSize] = self.columns[x*self.fieldLen+y].getEdgeImg()
        return ret

    def work(self):
        sensordata = self.sensor.getOutput()
        self.data = cv2.resize(sensordata, dsize=(self.fieldSize*self.fieldLen,self.fieldSize*self.fieldLen), interpolation=cv2.INTER_LINEAR)
        #self.data = cv2.Canny(image.astype(np.uint8), 100, 200)

    def getImg(self):
        return self.data