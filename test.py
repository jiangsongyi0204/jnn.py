import cv2
import numpy as np
import csv
from model1.sensor import Sensor
from model1.feature import Feature
from model1.column import Column

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)

sensor = Sensor('EdgeSensor')
column = Column('Column',sensor)
column1 = Column('Column1',column)

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.read(image)
    column.run()
    column1.run()
    cv2.imshow('Img',sensor.getSensorImg())
    cv2.imshow('Features',column.getFeaturesImg())
    cv2.imshow('Column output',column.getFeatureMapImg())
    cv2.imshow('Features1',column1.getFeaturesImg())
    cv2.imshow('Column1 output',column1.getFeatureMapImg())
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows()