import random
import numpy as np
import math
from model.featuremcell import FeatureMCell
from lib.helper import Helper

'''
Feature Column of the JNN. 
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
        self.vMap = []                  #Feature map value
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
                #If fmc not connected to the MAPS, Set Connected
                if (fmc.isConnected == False):
                    fmc.isConnected = True
                    self.connectFCC(fmc)
            else:
                fmc.run()
        if sum > self.fmcSize*0.9:
            self.isStable = True
            #Delete inactive fmcs
            #TODO

    def connectFCC(self, fmc):
        self.fMap.append(fmc)

    def output(self):
        step = int(self.inputField.size/self.fmcSize)
        for fmc in self.fMap:
            active = []
            for i in range(0,self.inputField.size,step):
                active_value = 0
                for link in fmc.sensorLinks:
                    if i+link.pos < self.inputField.size:
                        if self.inputField.inputData[i+link.pos] == 1:
                            active_value = active_value + 1
                if active_value < len(fmc.sensorLinks)*0.8:
                    active.append(0)
                else:
                    active.append(1)
            self.vMap.append(active)
            print(fmc.name)
            print(active)

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
    
    def getFeatureMap(self,activeonly=False):      
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