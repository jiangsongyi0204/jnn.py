import random
import numpy as np
import math
from model.featuremcell import FeatureMCell
from model.link import Link
from lib.helper import Helper
from time import gmtime, strftime
'''
Feature Column of the JNN. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~      Features MAP         ~
~ A:00101010101010101010101 ~
~ B:10010100010101000101110 ~
~ C:01010001011100010101001 ~
~ D:00101010101000101010101 ~
~ ......................... ~
~ X:00011010010101010010101 ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            | 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~        Features           ~
~   A,B,C,D,E,F,G,H,I,J,..  ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~       INPUT FIELD         ~
~ 0010100100101010001010101 ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
class FeatureColumn:

    def __init__(self, name, inputField, shouldInit=True):
        self.name = name                #Name of this Feature Column
        self.inputField = inputField    #Input field of this Feature Column
        self.fmcs = []                  #FMCs
        self.fmcSize = 100              #TODO: should related to inputField size
        self.isStable = False           #is this fmc stable, if ture this fcc will stop learning  
        self.inputData = []             #Output of this Feature Column , be the input Data to up layers
        self.fMap = []                  #Feature map
        if shouldInit:
            self.initFMC()              #Init the fmcs
    
    def initFMC(self):
        for i in range(0,self.fmcSize):
            fmc = FeatureMCell('FMC'+str(i), self.inputField, self)
            self.fmcs.append(fmc)

    def run(self):
        sum  = 0
        for fmc in self.fmcs:
            if fmc.isFixed:
                sum = sum + 1
            else:
                fmc.run()
        if sum > self.fmcSize*0.8:
            self.isStable = True
            #Delete inactive fmcs
            #TODO

    def makeFMap(self):
        self.fMap = []
        step = int(self.inputField.size/self.fmcSize)
        for fmc in self.fmcs:
            active = []
            if fmc.isFixed:
                active = [self.featureMatch(fmc,i*step) for i in range(0,self.fmcSize)]
                #active = [random.uniform(0, 1)*10 for i in range(0,self.fmcSize)]
            else:
                active = [0.0 for i in range(0,self.fmcSize)]
            self.fMap.append(active)

    def featureMatch(self,fmc,pos):
        ret = 0.0
        active_value = 0.0
        for link in fmc.sensorLinks:
            if pos+link.pos < self.inputField.size:
                if self.inputField.inputData[pos+link.pos] == 1.0:
                    active_value = active_value + 1.0

        if active_value > len(fmc.sensorLinks)*0.9:
            ret = 10.0
        return ret

    def makeInputData(self):
        self.inputData = []
        for fmc in self.fmcs:
            if fmc.isActive and fmc.isFixed:
                self.inputData.append(1.0)
            else:
                self.inputData.append(0.0)

    def getPredictFmc(self):
        predMap = [fmc.isNextActiveScore for fmc in self.fmcs]
        sortindex = np.argsort(predMap)
        sortindex = sortindex[:90]
        for idx in sortindex:
            predMap[idx] = 0.0
        #print(predMap)
        x = int(np.sqrt(self.fmcSize))
        return np.reshape(predMap,(x,x))
    
    def getFeaturesImg(self,activeonly=False):      
        fmcs = self.fmcs
        w = np.sqrt(self.fmcSize)
        rowa = []
        ret = []
        for i in range(0,self.fmcSize):
            if i % w == 0:
                if i > 0:
                    if i == w:
                        ret = rowa
                    else:
                        ret = np.concatenate((ret, rowa), axis=0) 
                rowa = fmcs[i].getFeatureImg(True,activeonly)
            else:
                rowa = np.concatenate((rowa, fmcs[i].getFeatureImg(True,activeonly)), axis=1)
        return ret
    
    def getFeatureMapImg(self):
        self.makeFMap()
        return np.array(self.fMap)
    
    def save(self):
        fileName = 'FC_v'+strftime("%Y%m%d_%H%M%S", gmtime())+'.txt'
        path_w = 'data/model/' + fileName
        for fmc in self.fmcs:
            with open(path_w, mode='a') as f:
                for link in fmc.sensorLinks:
                    f.write(str(link.pos))
                    f.write('^')
                    f.write(str(link.weight))
                    f.write(':')
                f.write('\n')
    
    def importFmc(self,modelName):
        path_r = 'data/model/' + modelName
        f = open(path_r, 'r')
        self.fmcs = []
        for line in f.readlines():
            links = line.rstrip('\r\n')[:-1].split(':')
            fmc = FeatureMCell('FMC',self.inputField,self,False)
            fmc.isFixed = True
            for link in links:
                l = link.split('^')
                linkObj = Link('L',self.inputField, int(l[0]), fmc, float(l[1]))
                fmc.sensorLinks.append(linkObj)
            self.fmcs.append(fmc)
        

