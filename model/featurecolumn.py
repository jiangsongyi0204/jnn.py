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
        self.isStable = False
        self.outputData = []
        self.initFMC()
    
    def initFMC(self):
        for i in range(0,self.fmc_num):
            fmc = FeatureMCell('FMC'+str(i), self.sensor, self)
            self.fmcs.append(fmc)

    def run(self):
        sum  = 0
        for fmc in self.fmcs:
            fmc.run()
            fmc.debug()
            if fmc.isFixed:
                sum = sum + 1
        if sum > self.fmc_num*0.5:
            self.isStable = True

        #self.output()
        '''
        for fmc in self.#fmcs:
            fmc.learnSequence()
        for fmc in self.fmcs:
            fmc.predict()

        #for fmc in self.fmcs:
        #    fmc.debug()
        '''

    def output(self):
        self.outputData = []
        for i in range(0,self.sensor.size):
            for fmc in self.fmcs:
                self.outputData.append(fmc.scanMap[i])
        
        print(self.outputData)
            

    def getPredictFmc(self):
        predMap = [fmc.isNextActiveScore for fmc in self.fmcs]
        sortindex = np.argsort(predMap)
        sortindex = sortindex[:90]
        for idx in sortindex:
            predMap[idx] = 0.0
        #print(predMap)
        x = int(np.sqrt(self.fmc_num))
        return np.reshape(predMap,(x,x))
    
    def getFeatureMap(self):       
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
                rowa = fmcs[i].getFeatureImg(border=True)
            else:
                rowa = np.concatenate((rowa, fmcs[i].getFeatureImg(border=True)), axis=1)
            
        return ret