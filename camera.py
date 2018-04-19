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

    sensor = Sensor('EdgeSensor', 100)
    fc = FeatureColumn('FC',sensor)

    while True:
        ret, image = capture.read()
        if ret == False:
            continue
        cv2.imshow("Original", image)
        sensor.readImage(image)
        cv2.imshow('Edges',cv2.resize(sensor.getSensorImg(),(200,200)))
        fc.run()
        #################
        cv2.imshow('Feature Map',cv2.resize(fc.getFeatureMap(),(500,500)))
        cv2.imshow('Predict Map',cv2.resize(fc.getPredictFmc(),(200,200)))
        
        #cv2.imshow('Feature Map', fc.getFeatureMap(srt=True))
        
        if cv2.waitKey(33) >= 0:
            break
    
    cv2.destroyAllWindows()