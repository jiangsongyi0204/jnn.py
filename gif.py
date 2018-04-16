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

sensor = Sensor('EdgeSensor',100)
fc = FeatureColumn('FC',sensor)

for image in imgs:
    cv2.waitKey(1)
    cv2.imshow("Original", image)
    size = 50
    resize_image = cv2.resize(image,(size,size)) 
    edged_image = cv2.Canny(resize_image,100,150)
    edged_image = cv2.resize(edged_image,(size,size))
    cv2.imshow('Edges',cv2.resize(edged_image,(200,200)))

    #################
    inputData = ''
    for x in range(size):
        for y in range(size):
            if edged_image[y,x] > 0:
                inputData += '1'
            else:
                inputData += '0'

    sensor.scan(inputData)
    fc.run()
    #cv2.imshow('Feature1',cv2.resize(fc.getSortedFMC()[0].getFeatureImg(),(200,200)))
    #################

fcmx = fc.getSortedFMC()
Helper.draw(fcmx)

cv2.destroyAllWindows()
