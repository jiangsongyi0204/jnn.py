import cv2
import numpy as np
import csv
from model2.sensor import Sensor
from model2.vision import Vision

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)

sensor = Sensor('EdgeSensor')
vision = Vision('Vision',sensor)

win1 = "Image"
cv2.namedWindow(win1)
cv2.moveWindow(win1, 40,30)

win2 = "Features"
cv2.namedWindow(win2)
cv2.moveWindow(win2, 180,30)

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.read(image)
    vision.run()
    cv2.imshow(win1, vision.getImg())    
    cv2.imshow(win2,vision.getColumnImg())
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows() 