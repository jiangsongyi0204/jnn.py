import random
import numpy as np
import math
from lib.helper import Helper
from model.link import Link

class FeatureMCell:

    def __init__(self, name, sensor):
        self.name = name
        self.sensor = sensor
        self.links = []
        self.status = 0
        self.score = 0.0
        self.activeFrq = 0
        self.initLinks()
        self.predict = []
        self.activeHistory = []

    def initLinks(self):
        sensorSize = self.sensor.size
        posarr = [m for m in range(0,sensorSize)]
        picksize = random.randrange(round(sensorSize*0.07), round(sensorSize*0.1))
        linksPos = random.sample(posarr,picksize)
        for idx, pos in enumerate(linksPos):
            link = Link('L'+str(idx),self.sensor,pos,self)
            self.links.append(link)
    
    def run(self):
        sum = 0.0
        for link in self.links:
            sum = sum + link.weight * self.sensor.inputData[link.pos]
        self.score = sum

        #20% links active -> featuremcell active
        if self.score > len(self.links)*0.2:
            self.activeHistory.append('1')
            self.activeFrq += 1
            for link in self.links:
                if self.sensor.inputData[link.pos] == 0:
                    link.downWeight()
                else:
                    link.upWeight()
        else:
            self.activeHistory.append('0')

        #remain the last 10 fire status
        if len(self.activeHistory) == 10:
            self.activeHistory.pop(0)

        #remove links
        newLinks = [item for item in self.links if item.weight > 0]
        self.links = newLinks

    def getFeatureImg(self,border=False):
        imgMap = [0 for m in range(0,self.sensor.size)]
        for link in self.links:
            if self.activeFrq > 0:
                imgMap[link.pos] = link.weight
        x = int(np.sqrt(self.sensor.size))
        fimg = np.reshape(imgMap,(x,x))
        if border == True:
            fimg = np.pad(fimg, 1, Helper.pad_with, padder=10)
        return fimg

    def debug(self,lev=0):
        d = self.name + ":" + str(self.activeFrq) + ":" + str(self.score) + "/" + str(len(self.links))+ ":" + ''.join(self.activeHistory) + ":"
        if lev>0:
            for link in self.links:
                d = d + '[' + str(round(link.weight, 2)) + '|' + str(link.pos) + ']'
        print(d)