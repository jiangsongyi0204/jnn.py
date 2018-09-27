import cv2
import numpy as np
import csv
from model.sensor import Sensor
from model.featurecolumn import FeatureColumn

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)

sensor = Sensor('EdgeSensor', 784)
fc = FeatureColumn('FC',sensor)

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.readMnist(image)
    #cv2.imshow("Sensor Image", cv2.resize(sensor.getSensorImg(),(100,100)))
    print('['+str(idx)+'] learning '+str(row[:1]))
    fc.run()
    if idx % 100 ==0:
        cv2.imshow('Features',fc.getFeaturesImg())
    if cv2.waitKey(33) >= 0:
        break

fc.save()

cv2.destroyAllWindows()