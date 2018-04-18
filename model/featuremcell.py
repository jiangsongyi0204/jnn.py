import random
import numpy as np
import math
from lib.helper import Helper
from model.link import Link

class FeatureMCell:

    def __init__(self, name, sensor, fc):
        self.name = name
        self.sensor = sensor
        self.sensorLinks = []
        self.fmcLinks = []
        self.status = 0
        self.score = 0.0
        self.activeFrq = 0
        self.isActive = False
        self.isPreActive = False
        self.willActive = 0.0
        self.predict = []
        self.activeHistory = []
        self.fc = fc
        #init functions
        self.initSensorLinks()

    def initSensorLinks(self):
        sensorSize = self.sensor.size
        posarr = [m for m in range(0,sensorSize)]
        picksize = random.randrange(round(sensorSize*0.07), round(sensorSize*0.1))
        linksPos = random.sample(posarr,picksize)
        for idx, pos in enumerate(linksPos):
            link = Link('L'+str(idx),self.sensor,pos,self)
            self.sensorLinks.append(link)

    def initFMCLinks(self, fmcs):
        for idx,fmc in enumerate(fmcs):
            link = Link('L'+str(idx),self.sensor,0,fmc)
            self.fmcLinks.append(link)

    def run(self):
        sum = 0.0
        for link in self.sensorLinks:
            sum = sum + link.weight * self.sensor.inputData[link.pos]
        self.score = sum

        #20% links active -> featuremcell active
        self.isPreActive = self.isActive
        if self.score > len(self.sensorLinks)*0.2:
            self.isActive = True
            self.activeHistory.append('1')
            self.activeFrq += 1
            for link in self.sensorLinks:
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
        newLinks = [item for item in self.sensorLinks if item.weight > 0]
        self.sensorLinks = newLinks      

    def learnSequence(self):
        #loop fmc links 
        #todo
        if self.isPreActive and self.activeFrq>100:
            for link in self.fmcLinks:
                linkedFmc = link.featuremcell
                if linkedFmc.isActive:
                    link.upWeight()
                else:
                    link.downWeight()

    def getFeatureImg(self,border=False):
        imgMap = [0 for m in range(0,self.sensor.size)]
        for link in self.sensorLinks:
            if self.activeFrq > 0:
                imgMap[link.pos] = link.weight
        x = int(np.sqrt(self.sensor.size))
        fimg = np.reshape(imgMap,(x,x))
        if border == True:
            fimg = np.pad(fimg, 1, Helper.pad_with, padder=10)
        return fimg

    def debug(self,lev=0):
        d = self.name + ":" + str(self.activeFrq) + ":" + str(self.score) + "/" + str(len(self.sensorLinks))+ ":" + ''.join(self.activeHistory) + ":"
        if lev>0:
            for link in self.sensorLinks:
                d = d + '[' + str(round(link.weight, 2)) + '|' + str(link.pos) + ']'
        print(d)