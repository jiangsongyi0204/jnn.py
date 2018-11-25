import numpy as np
from model2.column import Column

class Vision:

    def __init__(self, name, sensor):
        self.name = name
        self.sensor = sensor
        self.x = 0
        self.y = 0
        self.z = 0
        self.window = 400
        self.column = []
        self.init()
    
    def init(self):
        for x in range(10):
            for y in range(10):
                column = Column(str(x)+"-"+str(y),self,x,y,True)
                self.column.append(column)

    def getOutput(self):
        ret = np.zeros((self.window,self.window))
        px,py = self.sensor.getShape()
        inputdata = self.sensor.getOutput()
        for x in range(0,min(self.window-1,px-self.x)):
            for y in range(0,min(self.window-1,py-self.y)):
                ret[x,y] = inputdata[x+self.x,y+self.y]
        return ret