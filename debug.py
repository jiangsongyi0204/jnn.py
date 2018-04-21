import cv2
import numpy as np
import imageio
from model.sensor import Sensor
from model.featuremcell import FeatureMCell
from model.featurecolumn import FeatureColumn
from lib.helper import Helper

# convert form RGB to BGR 
imgs = [
    [0,0,0,0,1,0,0,0,0,0,
     0,0,0,1,0,1,0,0,0,0,
     0,0,1,0,0,0,1,0,0,0,
     0,1,0,0,0,0,0,1,0,0,
     1,0,0,0,0,0,0,0,1,0,
     0,1,0,0,0,0,0,0,0,1,
     0,0,1,0,0,0,0,0,1,0,
     0,0,0,1,0,0,0,1,0,0,
     0,0,0,0,1,0,1,0,0,0,
     0,0,0,0,0,1,0,0,0,0,],
    [0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,
     0,0,1,1,1,1,1,1,0,0,
     0,1,0,0,1,1,0,0,1,0,
     1,0,0,0,1,1,0,0,0,1,
     0,0,0,0,1,1,0,0,0,0,
     0,0,0,0,1,1,0,0,0,0,
     0,0,0,0,1,1,0,0,0,0,
     0,0,0,0,1,1,0,0,0,0,
     0,0,0,0,1,1,0,0,0,0,],
    [0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,
     0,1,1,1,1,1,1,1,1,0,
     0,0,1,1,1,1,1,1,0,0,
     0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,1,1,0,0,0,0,
     0,0,0,0,1,1,0,0,0,0,],
    [0,0,1,0,0,0,0,0,0,0,
     0,0,1,0,0,0,0,0,0,0,
     0,0,1,0,0,0,0,0,0,0,
     0,0,1,1,1,1,1,1,0,0,
     0,0,0,0,0,0,0,0,1,0,
     0,0,0,0,0,0,0,0,0,1,
     0,0,0,0,0,0,0,0,1,0,
     0,0,0,0,0,0,0,1,0,0,
     0,0,0,0,0,0,1,0,0,0,
     0,0,0,0,0,1,0,0,0,0,],

]

sensor = Sensor('EdgeSensor', 100)
fc = FeatureColumn('FC',sensor,fmc_num=49)

i = 0
while True:
    image = np.reshape(imgs[i],(10,10))
    i = i + 1
    if i == len(imgs):
        i = 0
    cv2.imshow("Original", image)
    sensor.readData(imgs[i])
    fc.run()
    cv2.imshow('Feature Map',cv2.resize(fc.getFeatureMap(),(500,500)))
    #cv2.imshow('Feature Map', fc.getFeatureMap(srt=True))
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows()
