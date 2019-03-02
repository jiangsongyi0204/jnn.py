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
        self.neighbors = []
        self.img = []
        self.matchedImg = []
        self.edgeImg = []
        self.predictedImg = []
        
    def run(self):
        #Run input field
        self.inputField.run()
        #Init neighbors
        self.initNeighbors()
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
                #fixedFeature = 255*np.ones((self.inputField.getSize(), self.inputField.getSize()), dtype = int)
                #fixedFeature[1:-1,1:-1] = feature.getImg()[1:-1,1:-1]
                #self.matchedImg = fixedFeature
                self.matchedImg = feature.getImg()
                self.feature_matched = True
                self.feature_matched_map[i] = 1
                break
            else:
                self.feature_matched_map[i] = 0
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

    def predict(self):
        self.predictedImg = np.zeros((self.inputField.visionSize,self.inputField.visionSize))
        maxPre = 0
        for feature in self.features:
            neighborMatched = []
            for column in self.neighbors:
                if column is not None:
                    neighborMatched.append(column.feature_matched_map)
                else:
                    neighborMatched.append(np.zeros(self.features_max))
            neighborMatched = np.array(neighborMatched)
            feature.predicted = (feature.learningMap * neighborMatched > 0).sum()
            if (feature.predicted > maxPre):
                maxPre = feature.predicted
                borderFeature = 255*np.ones((self.inputField.getSize(), self.inputField.getSize()), dtype = int)
                borderFeature[1:-1,1:-1] = feature.getImg()[1:-1,1:-1]*10
                self.predictedImg = borderFeature

    def learn(self):
        for feature in self.features:
            feature.learn()       
        if (self.feature_matched == False) :
            self.feature_matched_score = 0
    
    def initNeighbors(self):
        if (len(self.neighbors) == 0):
            self.neighbors.append(self.vision.getColumnByPos(self.vx-1,self.vy-1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx,self.vy-1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx+1,self.vy-1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx-1,self.vy))
            self.neighbors.append(self.vision.getColumnByPos(self.vx+1,self.vy))
            self.neighbors.append(self.vision.getColumnByPos(self.vx-1,self.vy+1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx,self.vy+1))
            self.neighbors.append(self.vision.getColumnByPos(self.vx+1,self.vy+1))
        
    def getNeighbors(self):
        return self.neighbors

    def getImg(self):
        return self.img
    
    def getMatchedImg(self):
        return self.matchedImg

    def getEdgeImg(self):
        return self.edgeImg
    
    def getPredictedImg(self):
        return self.predictedImg + self.matchedImg


