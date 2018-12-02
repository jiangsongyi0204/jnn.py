import random
import numpy as np
import math
from lib.helper import Helper
from model2.link import Link
from model2.sensor import Sensor

class Feature:

    def __init__(self, name, inputField, column):
        self.name = name
        self.inputField = inputField
        self.column = column
        self.links = []
        self.minPos = 9999
        self.isFixed = False
        self.isActive = False
        self.isInited = False

    def init(self):
        self.links = []
        self.minPos = 9999
        self.isFixed = False
        self.isActive = False
        #init functions
        self.initLinks()
        self.isInited = True

    def initLinks(self):
        self.links = []
        pickSize = self.inputField.getLength() * self.inputField.getSize()
        posRange = [p for p in range(0,pickSize)]
        posPick = random.randrange(round(pickSize*0.01), round(pickSize*0.02))
        linksPos = random.sample(posRange,posPick)
        for idx, pos in enumerate(linksPos):
            x = int(pos/self.inputField.getSize())
            y = int(pos%self.inputField.getSize())
            if y < self.minPos:
                self.minPos = y
            link = Link('L'+str(idx),self,self.inputField,x,y)
            self.links.append(link)

    def run(self):
        #init
        if self.isInited == False:
            self.init()
        #execute
        if self.isFixed == False:

            sum = 0.0
            for link in self.links:
                sum = sum + link.weight * self.inputField.getData()[link.idx][link.pos]

            #score
            self.isActiveScore = sum / len(self.links)

            #10% links active > featuremcell active
            if sum > len(self.links)*0.1:
                self.isActive = True
            else:
                self.isActive = False

            #Not fixed
            if self.isActive == True:
                for link in self.links:
                        if self.inputField.getData()[link.idx][link.pos] == 0:
                            link.downWeight()
                        else:
                            link.upWeight()
            else:
                for link in self.links:
                    link.lostWeight()

            #remove links
            newLinks = [item for item in self.links if item.weight > 0]
            #TODO:if len(newLinks) < self.inputField.size*0.002:
            if len(newLinks) < 2:
                self.init()
            else:
                self.links = newLinks
                if (self.isFixed == False):
                    self.doFix()

    def doFix(self):
        w = 0.0
        for link in self.links:
            w += link.weight
        if w > len(self.links)*0.99:
            print('DEBUG: ' + self.name + ' is Fixed.')
            self.isFixed = True

    def getImg(self):
        size = self.inputField.getSize()
        ret = np.zeros((size,size))
        for link in self.links:
            ret[int(link.pos/size)][int(link.pos%size)] = 255.0 * link.weight            
        return ret