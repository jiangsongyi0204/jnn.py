import cv2
import numpy as np
import csv
from model.sensor import Sensor
from model.featuremcell import FeatureMCell
from model.featurecolumn import FeatureColumn
from lib.helper import Helper

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)

sensor = Sensor('EdgeSensor', 784)
fc = FeatureColumn('FC',sensor,fmc_num=100)

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.readMnist(image)
    cv2.imshow("Sensor Image", cv2.resize(sensor.getSensorImg(),(100,100)))
    fc.run()
    cv2.imshow('Feature Map',cv2.resize(fc.getFeatureMap(),(500,500)))
    cv2.imshow('Column Output',fc.getOutputImg())    
    if idx > 9990:
        cv2.imwrite("data\output\mnist\d_" + row[0] + ".jpg", fc.getOutputImg())        
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows()