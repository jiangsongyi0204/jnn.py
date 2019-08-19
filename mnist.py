import cv2
import numpy as np
import csv
from model19.column import Column

f = open('data\input\mnist\mnist_test.csv')
reader = csv.reader(f)
column = Column("Columen",27)
for idx,row in enumerate(reader):
    x = np.array(row[1:])
    y = x.astype(np.float)
    image = np.reshape(y,(28,28))
    image = cv2.resize(image, (27,27))
    cv2.imshow('Image',image)
    column.forword(image)
    if cv2.waitKey(33) >= 0:
        break

cv2.destroyAllWindows()