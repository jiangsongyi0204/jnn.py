import cv2
import numpy as np

if __name__=="__main__":
    capture = cv2.VideoCapture(0)  
    if capture.isOpened() is False:
        raise("IO Error")
    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
    f = open('data\image.txt', 'w')
    time = 0
    while True:
        time = time + 1;
        ret, image = capture.read()
        if ret == False:
            continue
        cv2.imshow("Original", image)
        size = 25
        resize_image = cv2.resize(image,(size,size)) 
        edged_image = cv2.Canny(resize_image,100,150)
        #edged_image = cv2.resize(edged_image,(size,size))
        cv2.imshow('Edges',edged_image)
        if time > 10:
            f.write("'")
            for x in range(size):
                for y in range(size):
                    if edged_image[y,x] > 0:
                        f.write(str(1))
                    else:
                        f.write(str(0))
                #f.write('\n')
            f.write("',\n") 
            time = 0
        if cv2.waitKey(33) >= 0:
            #cv2.imwrite("data\images\image.png", edged_image)
            break
    cv2.destroyAllWindows()
    f.close()