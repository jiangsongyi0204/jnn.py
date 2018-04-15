import cv2
import numpy as np
from model.sensor import Sensor
from model.featuremcell import FeatureMCell
import matplotlib.pyplot as plt

def draw(fc):
    fig, ax = plt.subplots(nrows=5, ncols=5)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=1, top=1, wspace=0.05, hspace=0.05)
    i = 0
    for row in ax:
        for col in row:
            col.imshow(fc[i].getFeatureImg())
            fc[i].debug(1)
            i=i+1
    plt.show()

if __name__=="__main__":
    capture = cv2.VideoCapture(0)  
    if capture.isOpened() is False:
        raise("IO Error")
    #cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

    sensor = Sensor('EdgeSensor',625)
    fc = []
    for i in range(0,200):
        fmc = FeatureMCell('FMC'+str(i),sensor)
        fc.append(fmc)
    while True:
        ret, image = capture.read()
        if ret == False:
            continue
        cv2.imshow("Original", image)
        size = 25
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
        time = 0
        for fmc in fc:
            sensor.scan(inputData)
            fmc.run()
            fmc.debug()
        #################
 
        if cv2.waitKey(33) >= 0:
            fcmx = sorted(fc, key=lambda x: x.activeFrq, reverse=True)
            draw(fcmx)
            break
    
    cv2.destroyAllWindows()
