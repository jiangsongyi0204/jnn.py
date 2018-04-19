import cv2
import numpy as np
import imageio
from model.sensor import Sensor
from model.featuremcell import FeatureMCell
from model.featurecolumn import FeatureColumn
from lib.helper import Helper

## Read the gif from disk to `RGB`s using `imageio.miread` 
gif = imageio.mimread('data\input\giphy.gif')

# convert form RGB to BGR 
imgs = [img for img in gif]

sensor = Sensor('EdgeSensor', 100)
fc = FeatureColumn('FC',sensor)

i = 0
while True:
    image = imgs[i]
    i = i + 1
    if i == len(imgs):
        i = 0
    cv2.imshow("Original", image)
    sensor.readImage(image)
    cv2.imshow('Edges',cv2.resize(sensor.getSensorImg(),(200,200)))
    fc.run()
    cv2.imshow('Feature Map',cv2.resize(fc.getFeatureMap(),(500,500)))
    #cv2.imshow('Feature Map', fc.getFeatureMap(srt=True))
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows()
