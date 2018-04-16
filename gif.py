import cv2
import numpy as np
from model.sensor import Sensor
from model.featuremcell import FeatureMCell
import matplotlib.pyplot as plt
import imageio

def draw(fc):
    fig, ax = plt.subplots(nrows=2, ncols=5)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=1, top=1, wspace=0.05, hspace=0.05)
    i = 0
    for row in ax:
        for col in row:
            col.imshow(fc[i].getFeatureImg())
            i=i+1
    plt.show()

## Read the gif from disk to `RGB`s using `imageio.miread` 
gif = imageio.mimread('data\input\giphy.gif')

# convert form RGB to BGR 
imgs = [img for img in gif]

sensor = Sensor('EdgeSensor',100)
fc = []
for i in range(0,100):
    fmc = FeatureMCell('FMC'+str(i),sensor)
    fc.append(fmc)

for image in imgs:
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

    for fmc in fc:
        sensor.scan(inputData)
        fmc.run()
        fmc.debug()
    #################
    
    cv2.waitKey(1)

fcmx = sorted(fc, key=lambda x: x.activeFrq, reverse=True)
draw(fcmx)

cv2.destroyAllWindows()
