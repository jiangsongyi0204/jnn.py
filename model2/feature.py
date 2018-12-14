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
        self.isFixed = False
        self.isActive = False
        self.isInited = False
        self.size = self.inputField.getSize()

    def init(self):
        self.links = []
        self.isFixed = False
        self.isActive = False
        #init functions
        self.initLinks()
        self.isInited = True

    def initLinks(self):
        self.links = np.random.uniform(low=0, high=0.7, size=(self.size,self.size))

    def run(self):
        #init
        if self.isInited == False:
            self.init()
        #execute
        if self.isFixed == False:

            sum = self.links * self.inputField.getData()

            #score
            self.isActiveScore = (sum > 0).sum()

            #10% links active > featuremcell active
            if self.isActiveScore > self.size*self.size*0.1:
                self.isActive = True
            else:
                self.isActive = False

            #justy links
            if (self.isFixed == False):
                justy = sum
                justy[justy > 0] = 0.01
                justy[justy == 0] = -0.01
                self.links = self.links + justy

                #if links value > 0.4 smaller than 2 then reinit
                if (self.links > 0.4).sum() < 2:
                    self.init()
                else:
                    if (self.links > 0.99).sum() > 14:
                        self.isFixed = True

    def getImg(self):          
        return self.links