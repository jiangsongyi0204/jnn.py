import random
import numpy as np
import math
from model2.feature import Feature
from model2.link import Link
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

    def __init__(self, name, vision, vx, vy, field, shouldInit=True):
        self.name = name                #Name of this Feature Column
        self.vision = vision       #Vision of this Feature Column
        self.vx = vx
        self.vy = vy
        self.inputField = field
        self.features = []              #Features
        self.featureSize = 25           #TODO: should related to inputField size
        self.isStable = False           #is this Column stable, if ture it will stop learning  
        if shouldInit:
            self.init()              #Init the features
    
    def init(self):
        for i in range(0,self.featureSize):
            feature = Feature('Feature'+str(i), self.inputField, self)
            self.features.append(feature)

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
    
    def getImg(self):      
        ret = []
        for i in range(0,self.featureSize):
            if len(ret) == 0: 
                ret = self.features[i].getImg()
            else:
                ret = np.concatenate((ret, self.features[i].getImg()), axis=1)
        return ret
    
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
        

