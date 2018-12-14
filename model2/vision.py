import numpy as np
from model2.column import Column
from model2.field import Field

class Vision:

    def __init__(self, name, sensor, size, w):
        self.name = name
        self.sensor = sensor
        self.x = 0
        self.y = 0
        self.z = 0
        self.w = w
        self.size = size
        self.window = self.w*self.size
        self.fields = []
        self.columns = []
        self.data = np.zeros((self.window,self.window))
        self.init()
    
    def init(self):
        for x in range(self.w):
            for y in range(self.w):
                field = Field(str(x)+"-"+str(y),self,x,y)
                self.fields.append(field)
                column = Column(str(x)+"-"+str(y),self,x,y,field,True)
                self.columns.append(column)

    def getData(self):
        return self.data

    def run(self):
        self.work()
        for column in self.columns:
            column.run()

    def getImg(self):      
        ret = []
        for column in self.columns:
            if len(ret) == 0: 
                ret = column.getImg()
            else:
                ret = np.concatenate((ret, column.getImg()), axis=0)
        return ret

    def work(self):
        px,py = self.sensor.getShape()
        inputdata = self.sensor.getOutput()
        for x in range(0,min(self.window-1,px-self.x)):
            for y in range(0,min(self.window-1,py-self.y)):
                self.data[x,y] = inputdata[x+self.x,y+self.y]