import cv2
import numpy as np
from model201902.sensor import Sensor
from model201902.feature import Feature

if __name__=="__main__":
    capture = cv2.VideoCapture(0)  
    if capture.isOpened() is False:
        raise("IO Error")
    #cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    sensor = Sensor('EdgeSensor')
    feature1 = Feature('F1',sensor,[[0,0,1],[0,1,0],[1,0,0]])
    feature2 = Feature('F2',sensor,[[0,0,0],[1,1,1],[0,0,0]])
    #feature3 = Feature('F3',sensor,[[0,1,0],[1,1,1],[0,1,0]])
    feature4 = Feature('F4',sensor,[[1,0,0],[0,1,0],[0,0,1]])
    feature5 = Feature('F5',sensor,[[0,1,0],[0,1,0],[0,1,0]])
    sensor.addFeature(feature1)
    sensor.addFeature(feature2)
    #sensor.addFeature(feature3)
    sensor.addFeature(feature4)
    sensor.addFeature(feature5)
    x = 200

    cv2.namedWindow("Sensor")
    cv2.moveWindow("Sensor", 40,530)

    while True:
        ret, image = capture.read()
        if ret == False:
            continue
        height, width, channels = image.shape
        sensor.read(image)
        cv2.imshow("World", sensor.world)
        sensor.run()
        cv2.imshow("Sensor", sensor.output)
        if cv2.waitKey(33) >= 0:
            break
    
    cv2.destroyAllWindows()