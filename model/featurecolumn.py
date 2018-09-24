import random
import numpy as np
import math
from model.featuremcell import FeatureMCell
from lib.helper import Helper

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

    def __init__(self, name, inputField):
        self.name = name                #Name of this Feature Column
        self.inputField = inputField    #Input field of this Feature Column
        self.fmcs = []                  #FMCs
        self.fmcSize = 100              #TODO: should related to inputField size
        self.isStable = False           #is this fmc stable, if ture this fcc will stop learning  
        self.inputData = []             #Output of this Feature Column , be the input Data to up layers
        self.fMap = []                  #Feature map
        self.initFMC()                  #Init the fmcs
    
    def initFMC(self):
        for i in range(0,self.fmcSize):
            fmc = FeatureMCell('FMC'+str(i), self.inputField, self)
            self.fmcs.append(fmc)

    def run(self):
        if type(self.inputField) is FeatureColumn:
            if self.inputField.isStable == False:
                return
        sum  = 0
        for fmc in self.fmcs:
            if fmc.isFixed:
                sum = sum + 1
            else:
                fmc.run()
        if sum > self.fmcSize*0.5:
            self.isStable = True
            #Delete inactive fmcs
            #TODO

    def makeFMap(self):
        step = int(self.inputField.size/self.fmcSize)
        for fmc in self.fmcs:
            active = []
            if fmc.isFixed:
                active = [self.featureMatch(fmc,i*step) for i in range(0,self.fmcSize)]
            else:
                active = [0 for i in range(0,self.fmcSize)]
            self.fMap.append(active)

    def featureMatch(self,fmc,pos):
        ret = 0
        active_value = 0
        for link in fmc.sensorLinks:
            if pos+link.pos < self.inputField.size:
                if self.inputField.inputData[pos+link.pos] == 1:
                    active_value = active_value + 1

        if active_value > len(fmc.sensorLinks)*0.8:
            ret = 1
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
        ret = []
        w = int(np.sqrt(self.fmcSize))
        rowa = []
        self.makeFMap()
        for i in range(0,self.fmcSize):
            if i % w == 0:
                if i > 0:
                    if i == w:
                        ret = rowa
                    else:
                        ret = np.concatenate((ret, rowa), axis=0) 
                rowa = np.reshape(self.fMap[i],(w,w))
            else:
                rowa = np.concatenate((rowa, np.reshape(self.fMap[i],(w,w))), axis=1)
        return ret
