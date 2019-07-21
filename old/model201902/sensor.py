import numpy as np
import cv2

class Sensor:

    def __init__(self, name):
        self.name = name
        self.size = 100
        self.world = np.zeros((400,400))
        self.loc = (0,0,400) #location and size
        self.location = np.ones((self.size,self.size)) 
        self.data = np.zeros((self.size,self.size))
        self.features = []
        self.output = None
    
    def read(self, img):        
        self.world = cv2.resize(img, dsize=(400,400), interpolation=cv2.INTER_LINEAR)
    
    def addFeature(self, feature):
        self.features.append(feature)

    def run(self):
        loc_x,loc_y,loc_len = self.loc
        self.data = cv2.resize(self.world[loc_x:loc_len,loc_y:loc_len], dsize=(self.size,self.size), interpolation=cv2.INTER_LINEAR)
        for feature in self.features:
            feature.match()
        self.output = self.location
        for feature in self.features:    
            self.output = np.concatenate((self.output, feature.output), axis=1)
        max_F = 0
        select_F = None
        for feature in self.features:    
            if feature.output.sum() > max_F:
                max_F = feature.output.sum()
                select_F = feature
        print(max_F,select_F.name)
        

