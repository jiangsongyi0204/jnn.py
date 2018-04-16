import cv2
import numpy as np
from model.sensor import Sensor
from model.featuremcell import FeatureMCell
from model.featurecolumn import FeatureColumn
from lib.helper import Helper

if __name__=="__main__":
    capture = cv2.VideoCapture(0)  
    if capture.isOpened() is False:
        raise("IO Error")
    #cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

    size = 30

    sensor = Sensor('EdgeSensor', int(size*size))
    fc = FeatureColumn('FC',sensor)

    while True:
        ret, image = capture.read()
        if ret == False:
            continue
        cv2.imshow("Original", image)
        resize_image = cv2.resize(image,(size,size)) 
        edged_image = cv2.Canny(resize_image,100,150)
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
        #################
        cv2.imshow('Feature Map',cv2.resize(fc.getFeatureMap(srt=True),(500,500)))
 
        if cv2.waitKey(33) >= 0:
            fcmx = fc.getSortedFMC()
            Helper.draw(fcmx)
            break
    
    cv2.destroyAllWindows()