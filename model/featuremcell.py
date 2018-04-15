import random
from model.link import Link
import numpy as np
import math

class FeatureMCell:

    def __init__(self, name, sensor):
        self.name = name
        self.sensor = sensor
        self.links = []
        self.status = 0
        self.score = 0.0
        self.activeFrq = 0

        sensorSize = self.sensor.size
        posarr = [m for m in range(0,sensorSize)]
        picksize = random.randrange(round(sensorSize*0.01), round(sensorSize*0.06))
        linksPos = random.sample(posarr,picksize)
        for idx, pos in enumerate(linksPos):
            link = Link('L'+str(idx),sensor,pos,self)
            self.links.append(link)
    
    def run(self):
        sum = 0.0
        for link in self.links:
            sum = sum + link.weight * self.sensor.inputData[link.pos]
        self.score = sum

        #20% links active -> featuremcell active
        if self.score > len(self.links)*0.2:
            self.activeFrq += 1
            for link in self.links:
                if self.sensor.inputData[link.pos] == 0:
                    link.downWeight()
                else:
                    link.upWeight()

    def getFeatureImg(self):
        imgMap = [0.0 for m in range(0,self.sensor.size)]
        for link in self.links:
            if self.activeFrq > 0:
                imgMap[link.pos] = link.weight
        x = int(np.sqrt(self.sensor.size))
        return np.reshape(imgMap,(x,x))

    def debug(self,lev=0):
        d = self.name + ":" + str(self.activeFrq) + ":" + str(self.score) + "/" + str(len(self.links))+ ":"
        if lev>0:
            for link in self.links:
                d = d + '[' + str(round(link.weight, 2)) + '|' + str(link.pos) + ']'
        print(d)