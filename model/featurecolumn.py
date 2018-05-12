import random
import numpy as np
import math
from model.featuremcell import FeatureMCell
from lib.helper import Helper

class FeatureColumn:

    def __init__(self, name, sensor, fmc_num = 100):
        self.name = name
        self.sensor = sensor
        self.fmcs = []
        self.fmc_num = fmc_num
        self.size = fmc_num
        self.isStable = False
        self.outputData = []
        self.inputData = []
        self.initFMC()
    
    def initFMC(self):
        for i in range(0,self.fmc_num):
            fmc = FeatureMCell('FMC'+str(i), self.sensor, self)
            self.fmcs.append(fmc)

    def run(self):
        if type(self.sensor) is FeatureColumn:
            if self.sensor.isStable == False:
                return
        sum  = 0
        for fmc in self.fmcs:
            fmc.run()
            #fmc.debug()
            if fmc.isFixed:
                sum = sum + 1
        if sum > self.fmc_num*0.5:
            self.isStable = True

        self.makeInputData()
        #self.output()
        #self.outputFmcs()
        '''
        for fmc in self.#fmcs:
            fmc.learnSequence()
        for fmc in self.fmcs:
            fmc.predict()

        #for fmc in self.fmcs:
        #    fmc.debug()
        '''

    def output(self):
        sortedFmc = sorted(self.fmcs, key=lambda x : x.isActiveScore * int(x.isFixed), reverse=True)
        self.outputData = []
        for fmc in sortedFmc:
            for i in range(0,self.sensor.size):
                self.outputData.append(fmc.scanMap[i])

    def makeInputData(self):
        self.inputData = []
        for fmc in self.fmcs:
            if fmc.isActive and fmc.isFixed:
                self.inputData.append(1.0)
            else:
                self.inputData.append(0.0)

    def getOutputImg(self):
        return np.reshape(self.outputData,(self.fmc_num,self.sensor.size))

    def getPredictFmc(self):
        predMap = [fmc.isNextActiveScore for fmc in self.fmcs]
        sortindex = np.argsort(predMap)
        sortindex = sortindex[:90]
        for idx in sortindex:
            predMap[idx] = 0.0
        #print(predMap)
        x = int(np.sqrt(self.fmc_num))
        return np.reshape(predMap,(x,x))
    
    def getFeatureMap(self,activeonly=False):      
        fmcs = self.fmcs
        w = np.sqrt(self.fmc_num)
        rowa = []
        ret = []
        for i in range(0,self.fmc_num):
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