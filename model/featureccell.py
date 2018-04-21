import random
import numpy as np
import math
from lib.helper import Helper
from model.link import Link

class FeatureCCell:

    def __init__(self, name, sensor, fmc):
        self.name = name
        self.sensor = sensor
        self.fmc = fmc
        self.sensorLinks = []
        self.fmcLinks = []

        self.isActive = False
        self.isActiveScore = 0.0
        self.isPreActive = False
        self.isPreActiveScore = 0.0        
        self.isNextActive = False
        self.isNextActiveScore = 0.0        

    def run(self):
        self.isPreActive = self.isActive
        self.isPreActiveScore = self.isActiveScore
        sum = 0.0
        for link in self.sensorLinks:
            sum = sum + link.weight * self.sensor.inputData[link.pos]

        #score
        self.isActiveScore = sum / len(self.sensorLinks)

        #20% links active -> featuremcell active
        if sum > len(self.sensorLinks)*0.2:
            self.isActive = True
        else:
            self.isActive = False

    def learnSequence(self):
        if self.isPreActive:
            for link in self.fmcLinks:
                fcc = link.featuremcell.child[link.pos]
                if fcc.isActive:
                    link.upWeight()
                else:
                    link.downWeight()

    def predict(self):
        sum = 0.0
        for link in self.fmcLinks:
            fcc = link.featuremcell.child[link.pos]
            if fcc.isActive:
                sum = sum + link.weight
        
        self.isNextActiveScore = sum

        if sum > len(self.fmcLinks)*0.2:
            self.isNextActive = True
        else:
            self.isNextActive = False

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
        d = self.name + ":"
        if lev>0:
            for link in self.sensorLinks:
                d = d + '[' + str(round(link.weight, 2)) + '|' + str(link.pos) + ']'
            for link in self.fmcLinks:
                d = d + '(' + link.name + ')'                
        print(d)