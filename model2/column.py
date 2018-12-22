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
        self.feature_matched_score = 0
        self.feature_matched_map = np.zeros(self.features_max)
        self.feature_learning = False
        self.img = []
        self.matchedImg = []
        self.edgeImg = []
        self.neighbors = []
    
    def run(self):
        #Run input field
        self.inputField.run()
        #Feature match
        self.matchedImg = np.zeros((self.inputField.visionSize,self.inputField.visionSize))
        for i,feature in enumerate(self.features):
            feature.run()
            self.edgeImg = feature.getEdgeImg()
            if (feature.isFixed == False):
                self.feature_learning = True
            else:
                self.feature_learning = False
                self.img = self.img + feature.getImg()
            if (feature.isMatched):
                fixedFeature = 2*np.ones((self.inputField.getSize(), self.inputField.getSize()), dtype = int)
                fixedFeature[1:-1,1:-1] = feature.getImg()[1:-1,1:-1]
                self.matchedImg = fixedFeature
                self.feature_matched = True
                self.feature_matched_map[i] = 1
                break
            else :
                self.feature_matched = False
        #If no feature matched
        if (self.feature_matched == False and self.feature_learning == False and len(self.features) < self.features_max):
            feature = Feature(self.name+'Feature'+str(len(self.features)), self.inputField, self)
            self.features.append(feature)
            self.feature_learning = True
            feature.run()
            if (len(self.features)==1):
                self.img = np.zeros((self.inputField.visionSize,self.inputField.visionSize))
            self.edgeImg = feature.getEdgeImg()

    def learn(self):
        if (self.feature_matched):
            for feature in self.features:
                feature.learn()
            print(self.name,self.feature_matched_score)
        else:
            self.feature_matched_score = 0

    def getNeighbors(self):
        if (len(self.neighbors) == 0):
            self.neighbors.append(self.vision.getColumnByPos(self.vx-1,self.vy-1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx,self.vy-1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx+1,self.vy-1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx-1,self.vy))
            self.neighbors.append(self.vision.getColumnByPos(self.vx+1,self.vy))
            self.neighbors.append(self.vision.getColumnByPos(self.vx-1,self.vy+1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx,self.vy+1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx+1,self.vy+1))
        return self.neighbors

    def getImg(self):
        return self.img
    
    def getMatchedImg(self):
        return self.matchedImg

    def getEdgeImg(self):
        return self.edgeImg


