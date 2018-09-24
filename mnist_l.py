import cv2
import numpy as np
import csv
from model.sensor import Sensor
from model.featurecolumn import FeatureColumn

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)

sensor = Sensor('EdgeSensor', 784)
fc = FeatureColumn('FC',sensor,False)
fc.importFmc('FC_v20180924_041816.txt')

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.readMnist(image)
    cv2.imshow("Sensor Image", cv2.resize(sensor.getSensorImg(),(100,100)))
    fc.run()
    cv2.imshow('Features',fc.getFeaturesImg())
    if (row[0] == '1'):
        cv2.imshow('FM',fc.getFeatureMapImg())
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows() 