import random
from model.featurecolumn import FeatureColumn
import numpy as np
import math

class JNN:

    def __init__(self, name):
        self.name = name
        self.fcs = []
    
    def appendFc(self, fc):
        self.fcs.append(fc)

    def save(self):
        #save jnn to data store
        print("save jnn to data store")