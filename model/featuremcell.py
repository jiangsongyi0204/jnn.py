import random
import numpy as np
import math
from lib.helper import Helper
from model.link import Link
from model.clink import CellLink
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
        #init functions
        self.initSensorLinks()

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

    def initFMCConnect(self, fmcs):
        self.fmcConnect = []
        for fmc in fmcs:
            self.fmcConnect.append(fmc)

    def run(self):
        self.isPreActive = self.isActive
        self.isPreActiveScore = self.isActiveScore

        if self.isFixed == True:
            if len(self.child)==0:
                self.debug(lev=1)
            isAct = False
            sum = 0
            for fcc in self.child:
                if fcc.isActive:
                    isAct = True
                    sum = sum + 1
            self.isActive = isAct
            self.isActiveScore = sum / len(self.child)
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
            if len(newLinks) < self.sensor.size*0.02:
                self.initV()
            else:
                self.sensorLinks = newLinks
                self.doFix()

    def doFix(self):
        w = 0.0
        for link in self.sensorLinks:
            w += link.weight
        
        if w > len(self.sensorLinks)*0.9:
            self.isFixed = True
            self.makeChild()

    def makeChild(self):
        self.child = []
        if self.isFixed:
            #1.link sensor
            sensorlinksSorted = sorted(self.sensorLinks, key=lambda x : x.pos)
            min_p = sensorlinksSorted[0].pos
            max_p = sensorlinksSorted[-1].pos
            range_l = self.sensor.size - max_p + min_p
            if range_l<10:
                self.initV()
                return 
            for i in range(0,range_l):
                fcc = FeatureCCell(self.name + '-C'+str(i), self.sensor, self)
                for idx,sl in enumerate(self.sensorLinks):
                    link = Link(fcc.name+'-l'+str(idx),self.sensor,sl.pos-min_p+i,None)
                    link.weight = 1.0
                    fcc.sensorLinks.append(link)
                self.child.append(fcc)

    def learnSequence(self):
        if self.isChildLinked:
            #if self.isPreActive:
            for fcc in self.child:
                fcc.learnSequence()
        else:
            self.linkChild()

    def linkChild(self):
        fc = self.fc
        if fc.isStable:
            fmcPos = Helper.pos_random_sample(fc.fmc_num,0.2,0.5)
            for pos in fmcPos:
                fmc_con = fc.fmcs[pos]
                if fmc_con.isFixed:
                    for fcc in self.child:
                        len_child_to = len(fmc_con.child)
                        linksPos_to = Helper.pos_random_sample(len_child_to,0.4,0.6)
                        for pos_to in linksPos_to:
                            linkFcc = fmc_con.child[pos_to] 
                            link = CellLink(fcc.name + '<->' + linkFcc.name, fcc, linkFcc)
                            fcc.fmcLinks.append(link)
            self.isChildLinked = True

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
        if lev>0:
            for link in self.sensorLinks:
                d = d + '[' + str(round(link.weight, 2)) + '|' + str(link.pos) + ']'
            for fcc in self.child:
                for link in fcc.fmcLinks:
                    d = d + "\n" + link.name + ':' + str(link.weight) + ':' + str(link.weight*link.featuremcell.child[link.pos].isActiveScore)
        print(d)