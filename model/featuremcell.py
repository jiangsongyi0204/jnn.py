import random
import numpy as np
import math
from lib.helper import Helper
from model.link import Link
from model.clink import CellLink
from model.fmclink import FMCLink
from model.featureccell import FeatureCCell

class FeatureMCell:

    def __init__(self, name, sensor, fc):
        self.name = name
        self.sensor = sensor
        self.fc = fc
        self.sensorLinks = []
        self.fmcConnect = []
        self.isFixed = False
        self.isActive = False
        self.isActiveScore = 0.0
        self.isPreActive = False
        self.isPreActiveScore = 0.0        
        self.isNextActive = False
        self.isNextActiveScore = 0.0
        #child cell 
        self.child = []
        self.isChildLinked = False
        #
        self.isFMCLinked = False
        #
        self.scanMap = []
        #init functions
        self.initSensorLinks()
        self.initScanMap()

    def initV(self):
        self.sensorLinks = []
        self.fmcConnect = []
        self.isFixed = False
        self.isActive = False
        self.isActiveScore = 0.0
        self.isPreActive = False
        self.isPreActiveScore = 0.0        
        self.isNextActive = False
        self.isNextActiveScore = 0.0
        #child cell 
        self.child = []
        self.isChildLinked = False
        #
        self.isFMCLinked = False
        #
        self.scanMap = []
        #init functions
        self.initSensorLinks()
        self.initScanMap()

    def initSensorLinks(self):
        self.sensorLinks = []
        sensorSize = self.sensor.size
        posarr = [m for m in range(0,sensorSize)]
        picksize = random.randrange(round(sensorSize*0.01), round(sensorSize*0.03))
        linksPos = random.sample(posarr,picksize)
        for idx, pos in enumerate(linksPos):
            link = Link('L'+str(idx),self.sensor,pos,self)
            self.sensorLinks.append(link)

    def initScanMap(self):
        self.scanMap = [0.0 for m in range(0,self.sensor.size)]

    def initFMCConnect(self, fmcs):
        self.fmcConnect = []
        for fmc in fmcs:
            self.fmcConnect.append(fmc)

    def run(self):
        
        self.isPreActive = self.isActive
        self.isPreActiveScore = self.isActiveScore

        if self.isFixed == True:
            self.scanAll()
            isAct = False
            sum = 0
            for v in self.scanMap:
                if v>0:
                    isAct = True
                    sum = sum + 1
            self.isActive = isAct
            self.isActiveScore = sum / len(self.scanMap)
        else:
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

            #Not fixed
            if self.isActive == True:
                for link in self.sensorLinks:
                        if self.sensor.inputData[link.pos] == 0:
                            link.downWeight()
                        else:
                            link.upWeight()    

            #remove links
            newLinks = [item for item in self.sensorLinks if item.weight > 0]
            if len(newLinks) < self.sensor.size*0.002:
                self.initV()
            else:
                self.sensorLinks = newLinks
                self.doFix()

    def doFix(self):
        w = 0.0
        for link in self.sensorLinks:
            w += link.weight
        
        if w > len(self.sensorLinks)*0.99:
            self.isFixed = True

    def scanAll(self):
        if self.isFixed:
            self.initScanMap()
            #1.link sensor
            sensorlinksSorted = sorted(self.sensorLinks, key=lambda x : x.pos)
            min_p = sensorlinksSorted[0].pos
            max_p = sensorlinksSorted[-1].pos
            range_l = self.sensor.size - max_p + min_p
            sl_len = len(self.sensorLinks)
            for i in range(0,range_l):
                sum = 0
                for sl in self.sensorLinks:
                    inD = self.sensor.inputData[sl.pos-min_p+i]
                    if inD == 0:
                        sum = sum + 1
                if sum > sl_len*0.9:
                    self.scanMap[i] = 1.0    

    def learnSequence(self):
        if self.isFMCLinked:
            if self.isPreActive:
                for fcc in self.child:
                    fcc.learnSequence()
        else:
            self.linkFMC()

    def linkChild(self):
        if self.fc.isStable:
            fixedFmc = [fmc for fmc in self.fc.fmcs if fmc.isFixed == True]
            selectedFixedFmc = Helper.arr_random_sample(fixedFmc,0.2,0.5)
            for fmc_con in selectedFixedFmc:
                for fcc in self.child:
                    len_child_to = len(fmc_con.child)
                    linksPos_to = Helper.pos_random_sample(len_child_to,0.4,0.6)
                    for pos_to in linksPos_to:
                        linkFcc = fmc_con.child[pos_to] 
                        link = CellLink(fcc.name + '<->' + linkFcc.name, fcc, linkFcc)
                        fcc.fmcLinks.append(link)
            self.isChildLinked = True

    def linkFMC(self):
        if self.fc.isStable:
            fixedFmc = [fmc for fmc in self.fc.fmcs if fmc.isFixed == True]
            for fmc in fixedFmc:
                link =  FMCLink(self.name + '<->' + fmc.name, self, fmc)  

    def predict(self):
        score = 0
        sum = 0
        for fcc in self.child:
            fcc.predict()
            if fcc.isNextActive:
                sum = sum + 1
            score = score + fcc.isNextActiveScore
        self.isNextActiveScore = score
        if sum > len(self.child)*0.02:
            self.isNextActive = True
        else:
            self.isNextActive = False

    def getFeatureImg(self,border=False):
        imgMap = [0.0 for m in range(0,self.sensor.size)]
        for link in self.sensorLinks:
            if self.isFixed:
                imgMap[link.pos] = link.weight * 10
            else:
                imgMap[link.pos] = link.weight
        x = int(np.sqrt(self.sensor.size))
        fimg = np.reshape(imgMap,(x,x))
        if border == True:
            fimg = np.pad(fimg, 1, Helper.pad_with, padder=255)
        return fimg

    def debug(self,lev=0):
        d = self.name + ":" + str(self.isActiveScore) + ":" + str(self.isNextActiveScore) + ":" + str(self.isChildLinked)
        if lev>0 and self.isChildLinked:
            for link in self.sensorLinks:
                d = d + '[' + str(round(link.weight, 2)) + '|' + str(link.pos) + ']'
            for fcc in self.child:
                fcc.debug(lev=lev)
        print(d)