import random
import numpy as np
import math
import cv2
from lib.helper import Helper
from model2.link import Link
from model2.sensor import Sensor

class Feature:

    def __init__(self, name, inputField, column):
        self.name = name
        self.inputField = inputField
        self.column = column
        self.links = []
        self.edge = []
        self.isFixed = False
        self.isMatched = False
        self.isInited = False
        self.predicted = 0
        self.learningMap = np.zeros((8,self.column.features_max))
        self.size = self.inputField.getSize()
    
    def init(self):
        self.isFixed = False
        self.isMatched = False
        #self.links = np.random.uniform(low=0, high=0.7, size=(self.size,self.size))
        self.links = np.full((self.size,self.size),0.5)
        self.isInited = True

    def run(self):
        #init
        if self.isInited == False:
            self.init()
        #edge feature 
        self.edge = cv2.Canny(self.inputField.getData().astype(np.uint8), 100, 200)
        self.edge[self.edge > 0] = 1
        self.edge = self.edge.astype(np.float32)
        #If no input break execute
        #if (self.edge > 0).sum() < self.size*0.02:
        #    return
        #execute
        if self.isFixed == False:
            matched = self.links * self.edge
            #tweak links
            matched[matched > 0] = 0.08
            matched[matched == 0] = -0.01
            self.links = self.links + matched
            #if links value > 0.4 smaller than 2 then reinit
            if (self.links > 0.5).sum() < self.size*self.size*0.01:
                self.init()
            else:
                if (self.links > 0.95).sum() > self.size*self.size*0.01:
                    self.isFixed = True
                    self.links[self.links > 0.95] = 1
                    self.links[self.links < 0.95] = 0
                    self.links = self.links.astype(np.float32)
        else:
            tLinks = np.copy(self.links)
            tLinks[tLinks == 0] = 2
            tedge = np.copy(self.edge)
            tedge[tedge == 0] = 3 
            matched = (tLinks == tedge).astype(np.float32)
            size = min(self.links.sum(),self.edge.sum())
            if (matched.sum() > size*0.2 ):
                self.isMatched = True
            else:
                self.isMatched = False

    def learn(self):
        for i,column in enumerate(self.column.getNeighbors()):
            if (column is not None):
                if (column.feature_matched):
                    adjust = column.feature_matched_map
                    if (self.isMatched):                    
                        adjust[adjust == 0] = -0.01
                        adjust[adjust > 0] = 0.02
                    else: 
                        adjust[adjust > 0] = -0.001
                    self.learningMap[i] = self.learningMap[i] + adjust
                    #self.column.feature_matched_score = self.column.feature_matched_score + (column.feature_matched_map > 0).sum()
        self.learningMap[self.learningMap < 0] = 0
        self.learningMap[self.learningMap > 1] = 1

    def getImg(self):
        return self.links
    
    def getEdgeImg(self):
        return self.edge