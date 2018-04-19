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
        self.isFixed = False
        self.score = 0.0
        self.isActive = False
        self.isPreActive = False
        self.willActive = 0.0
        self.predict = []
        self.fc = fc
        self.child = []
        #init functions
        self.initSensorLinks()

    def initSensorLinks(self):
        self.sensorLinks = []
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
        #
        self.isPreActive = self.isActive
        #
        sum = 0.0
        for link in self.sensorLinks:
            sum = sum + link.weight * self.sensor.inputData[link.pos]
        self.score = sum

        #20% links active -> featuremcell active
        if self.score > len(self.sensorLinks)*0.2:
            self.isActive = True
        else:
            self.isActive = False

        #Not fixed
        if self.isFixed == False:
            if self.isActive == True:
                for link in self.sensorLinks:
                        if self.sensor.inputData[link.pos] == 0:
                            link.downWeight()
                        else:
                            link.upWeight()    

            #remove links
            newLinks = [item for item in self.sensorLinks if item.weight > 0]
            if len(newLinks) < self.sensor.size*0.02:
                self.initSensorLinks()
            else:
                self.sensorLinks = newLinks
                self.doFix()

    def doFix(self):
        w = 0.0
        for link in self.sensorLinks:
            w += link.weight
        
        if w > len(self.sensorLinks)*0.9:
            self.isFixed = True

    def output(self):
        if self.isFixed:
            sensorlinksSorted = sorted(self.sensorLinks, key=lambda x : x.pos)
            min_p = sensorlinksSorted[0].pos
            max_p = sensorlinksSorted[-1].pos
            range_l = self.sensor.size - max_p - min_p 
            for i in range(0,range_l):
                sum = 0.0
                for sl in self.sensorLinks:
                    sum = sum + self.sensor.inputData[sl.pos-min_p+i]*sl.weight
                if sum > len(self.sensorLinks)*0.7:
                    self.child.append(0)
                else:
                    self.child.append(1)
        return self.child

    def learnSequence(self):
        #loop fmc links 
        #todo
        if self.isPreActive and self.isFixed:
            for link in self.fmcLinks:
                linkedFmc = link.featuremcell
                if linkedFmc.isActive:
                    link.upWeight()
                else:
                    link.downWeight()

    def getFeatureImg(self,border=False):
        imgMap = [0 for m in range(0,self.sensor.size)]
        for link in self.sensorLinks:
            imgMap[link.pos] = link.weight
        x = int(np.sqrt(self.sensor.size))
        fimg = np.reshape(imgMap,(x,x))
        if border == True:
            fimg = np.pad(fimg, 1, Helper.pad_with, padder=10)
        return fimg

    def debug(self,lev=0):
        d = self.name + ":" + str(self.score) + "/" + str(len(self.sensorLinks))+ ":"
        if lev>0:
            for link in self.sensorLinks:
                d = d + '[' + str(round(link.weight, 2)) + '|' + str(link.pos) + ']'
        print(d)