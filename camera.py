import cv2
import numpy as np

if __name__=="__main__":
    capture = cv2.VideoCapture(0)  
    if capture.isOpened() is False:
        raise("IO Error")
    while True:
        ret, image = capture.read()
        frame = cv2.resize(image, dsize=(200,200), interpolation=cv2.INTER_LINEAR)
        if ret == False:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        lower_red = np.array([0,20,50])
        upper_red = np.array([50,100,100])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        lower_red1 = np.array([50,20,50])
        upper_red1 = np.array([70,100,100])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        res1 = cv2.bitwise_and(frame,frame, mask= mask1)

        lower_red2 = np.array([70,20,50])
        upper_red2 = np.array([150,100,100])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        res2 = cv2.bitwise_and(frame,frame, mask= mask2)

        lower_red2 = np.array([70,20,50])
        upper_red2 = np.array([150,100,100])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        res2 = cv2.bitwise_and(frame,frame, mask= mask2)

        lower_red3 = np.array([150,20,50])
        upper_red3 = np.array([200,100,100])
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
        res3 = cv2.bitwise_and(frame,frame, mask= mask3)

        lower_red4 = np.array([200,20,50])
        upper_red4 = np.array([300,100,100])
        mask4 = cv2.inRange(hsv, lower_red4, upper_red4)
        res4 = cv2.bitwise_and(frame,frame, mask= mask4)

        lower_red5 = np.array([300,20,50])
        upper_red5 = np.array([360,100,100])
        mask5 = cv2.inRange(hsv, lower_red5, upper_red5)
        res5 = cv2.bitwise_and(frame,frame, mask= mask5)

        cv2.imshow('frame',frame)
        cv2.imshow('res',res)
        cv2.imshow('res1',res1)
        cv2.imshow('res2',res2)
        
        cv2.imshow('res3',res3)
        cv2.imshow('res4',res4)
        cv2.imshow('res5',res5)
        #height, width, channels = image.shape
        #result = cv2.Canny(image, 100, 200)
        #cv2.imshow("Sensor", image)


        if cv2.waitKey(33) >= 0:
            break
    
    cv2.destroyAllWindows()
    capture.release()