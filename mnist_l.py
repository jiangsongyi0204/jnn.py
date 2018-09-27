import cv2
import numpy as np
import csv
from model.sensor import Sensor
from model.featurecolumn import FeatureColumn

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)

sensor = Sensor('EdgeSensor', 784)
fc = FeatureColumn('FC',sensor,False)
fc.importFmc('FC_v20180927_130749.txt')

win1 = "Image"
cv2.namedWindow(win1)
cv2.moveWindow(win1, 40,30)

win2 = "Features"
cv2.namedWindow(win2)
cv2.moveWindow(win2, 180,30)

win3 = "Output"
cv2.namedWindow(win3)
cv2.moveWindow(win3, 500,30)

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.readMnist(image)
    cv2.imshow(win1, cv2.resize(sensor.getSensorImg(),(100,100)))
    fc.run()
    cv2.imshow(win2,fc.getFeaturesImg())
    #if (row[0] == '1'):
    cv2.imshow(win3,fc.getFeatureMapImg())
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows() 