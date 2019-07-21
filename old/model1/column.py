import random
import numpy as np
import math
from model1.feature import Feature
from model1.link import Link
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
class Column:

    def __init__(self, name, inputField, shouldInit=True):
        self.name = name                #Name of this Feature Column
        self.inputField = inputField    #Input field of this Feature Column
        self.features = []              #Features
        self.featureSize = 100          #TODO: should related to inputField size
        self.isStable = False           #is this Column stable, if ture it will stop learning  
        self.fMap = []                  #Feature map
        if shouldInit:
            self.init()              #Init the features
    
    def init(self):
        for i in range(0,self.featureSize):
            feature = Feature('Feature'+str(i), self.inputField, self)
            self.features.append(feature)

    def getData(self):
        return self.fMap

    def getLength(self):
        return len(self.fMap)
    
    def getSize(self):
        return len(self.fMap[0])

    def run(self):
        sum  = 0
        for feature in self.features:
            if feature.isFixed:
                sum = sum + 1
            else:
                feature.run()
        if sum > self.featureSize*0.3:
            self.isStable = True
            #Delete inactive features
            #TODO
        self.makeFMap()

    def makeFMap(self):
        self.fMap = []
        step = int(self.inputField.getSize()/self.featureSize)
        for feature in self.features:
            active = []
            if feature.isFixed:
                active = [self.featureMatch(feature,i*step) for i in range(0,self.featureSize)]
            else:
                active = [0.0 for i in range(0,self.featureSize)]
            self.fMap.append(active)

    def featureMatch(self, feature, pos):
        ret = 0.0
        active_value = 0.0
        for link in feature.links:
            p = pos+(link.pos-feature.minPos) 
            if p < self.inputField.getSize():
                if self.inputField.getData()[link.idx][p] > 0.0:
                    active_value = active_value + 1.0
        if active_value > len(feature.links)*0.9:
            ret = 10.0
        return ret
    
    def getFeaturesImg(self):      
        ret = []
        for i in range(0,self.featureSize):
            if len(ret) == 0: 
                ret = self.features[i].getFeatureImg()
            else:
                ret = np.concatenate((ret, self.features[i].getFeatureImg()), axis=1)
        return ret
    
    def getFeatureMapImg(self):
        return np.array(self.fMap)
    
    def save(self):
        fileName = 'FC_v'+strftime("%Y%m%d_%H%M%S", gmtime())+'.txt'
        path_w = 'data/model/' + fileName
        for feature in self.features:
            with open(path_w, mode='a') as f:
                for link in feature.links:
                    f.write(str(link.pos))
                    f.write('^')
                    f.write(str(link.weight))
                    f.write(':')
                f.write('\n')
    
    def importfeature(self,modelName):
        path_r = 'data/model/' + modelName
        f = open(path_r, 'r')
        self.features = []
        for line in f.readlines():
            links = line.rstrip('\r\n')[:-1].split(':')
            feature = Feature('feature',self.inputField,self)
            feature.isFixed = True
            feature.isInit = True
            for link in links:
                l = link.split('^')
                linkObj = Link('L',self.inputField, int(l[0]), feature, float(l[1]))
                feature.links.append(linkObj)
            self.features.append(feature)
        

