import random
from model.link import Link
import numpy as np
import math

class FeatureColumn:

    def __init__(self, name):
        self.name = name
        self.fmcs = []
    
    def initFmcs(self):
        f = self.fmcs