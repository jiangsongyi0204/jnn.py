import cv2
import numpy as np
import csv
from model1.sensor import Sensor
from model1.feature import Feature
from model1.column import Column

win_image = "Image"
cv2.namedWindow(win_image)
cv2.moveWindow(win_image, 10,0)

win_F1 = "Features1"
cv2.namedWindow(win_F1)
cv2.moveWindow(win_F1, 10,60)

win_F1_MAP = "M1"
cv2.namedWindow(win_F1_MAP)
cv2.moveWindow(win_F1_MAP, 10,400)

win_F2 = "Features2"
cv2.namedWindow(win_F2)
cv2.moveWindow(win_F2, 10,120)

win_F2_MAP = "M2"
cv2.namedWindow(win_F2_MAP)
cv2.moveWindow(win_F2_MAP, 200,400)

win_F3 = "Features3"
cv2.namedWindow(win_F3)
cv2.moveWindow(win_F3, 10,180)

win_F3_MAP = "M3"
cv2.namedWindow(win_F3_MAP)
cv2.moveWindow(win_F3_MAP, 400,400)

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)

sensor = Sensor('EdgeSensor')
column1 = Column('Column1',sensor)
column2 = Column('Column2',column1)
column3 = Column('Column3',column2)

for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    sensor.read(image)
    column1.run()
    column2.run()
    column3.run()
    cv2.imshow(win_image,sensor.getSensorImg())
    cv2.imshow(win_F1,column1.getFeaturesImg())
    cv2.imshow(win_F1_MAP,column1.getFeatureMapImg())
    cv2.imshow(win_F2,column2.getFeaturesImg())
    cv2.imshow(win_F2_MAP,column2.getFeatureMapImg())
    cv2.imshow(win_F3,column3.getFeaturesImg())
    cv2.imshow(win_F3_MAP,column3.getFeatureMapImg())
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows()