import cv2
import numpy as np
from model2.sensor import Sensor
from model2.vision import Vision
from lib.helper import Helper

if __name__=="__main__":
    capture = cv2.VideoCapture(0)  
    if capture.isOpened() is False:
        raise("IO Error")
    #cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    sensor = Sensor('EdgeSensor')
    vision = Vision('Vision',sensor)
    while True:
        ret, image = capture.read()
        if ret == False:
            continue
        height, width, channels = image.shape
        #cv2.imshow("Original", image)
        result = cv2.Canny(image, 100, 200)
        #cv2.imshow("Edge", result)
        sensor.read(image)
        vision.run()
        cv2.imshow("Vision", vision.getImg())
        cv2.imshow("Col Edge", vision.getColumnImg(3))
        cv2.imshow("Col Feature", vision.getColumnImg(1))
        cv2.imshow("Col Matched Feature", vision.getColumnImg(2))
        if cv2.waitKey(33) >= 0:
            break
    
    cv2.destroyAllWindows()