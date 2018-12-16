import random
import numpy as np
import math
from model2.feature import Feature
from model2.link import Link
from lib.helper import Helper
from time import gmtime, strftime

'''
TODO:
'''
class Column:

    def __init__(self, name, vision, vx, vy, field):
        self.name = name                #Name of this Feature Column
        self.vision = vision       #Vision of this Feature Column
        self.vx = vx
        self.vy = vy
        self.inputField = field
        self.features = []              #Features
        self.features_max = 1
        self.feature_matched = False
        self.feature_learning = False
        self.img = []
        self.matchedImg = []
        self.edgeImg = []
    
    def run(self):
        #Run input field
        self.inputField.run()
        #Feature match
        self.matchedImg = np.zeros((self.inputField.visionSize,self.inputField.visionSize))
        for feature in self.features:
            feature.run()
            self.img = feature.getImg()
            self.edgeImg = feature.getEdgeImg()
            if (feature.isFixed == False):
                self.feature_learning = True
            else:
                self.feature_learning = False
            if (feature.isMatched):
                self.matchedImg = feature.getImg()
                self.feature_matched = True
                break
            else :
                self.feature_matched = False
        #If no feature matched
        if (self.feature_matched == False and self.feature_learning == False and len(self.features) < self.features_max):
            feature = Feature(self.name+'Feature'+str(len(self.features)), self.inputField, self)
            self.features.append(feature)
            self.feature_learning = True
            feature.run()
            self.img = feature.getImg()
            self.edgeImg = feature.getEdgeImg()

    def getImg(self):
        return self.img
    
    def getMatchedImg(self):
        return self.matchedImg

    def getEdgeImg(self):
        return self.edgeImg


