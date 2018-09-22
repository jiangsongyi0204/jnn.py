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
fc = FeatureColumn('FC',sensor)
#fc1 = FeatureColumn('FC1',fc,fmc_num=100)
#fc2 = FeatureColumn('FC2',fc1,fmc_num=100)
#fc3 = FeatureColumn('FC3',fc2,fmc_num=100)

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.readMnist(image)
    cv2.imshow("Sensor Image", cv2.resize(sensor.getSensorImg(),(100,100)))
    fc.run()
    fc.output()
    #cv2.imshow('Feature Map 1',fc.getFeatureMap())
    #fc1.run()
    #cv2.imshow('Feature Map 2',fc1.getFeatureMap())
    #fc2.run()
    #cv2.imshow('Feature Map 3',fc2.getFeatureMap(True))
    #print(row[0]+":"+''.join(str(fc.inputData)))
    #fc3.run()
    #cv2.imshow('Feature Map 4',fc3.getFeatureMap(True))
    #u = fc.getOutputImg()
    #cv2.imshow('Column Output', fc.getOutputImg())
    #cv2.imwrite('data\output\mnist\d_' + row[0] + '.txt', u) 

    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows()