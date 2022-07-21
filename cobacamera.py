import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640) #ubah lebar video
cam.set(4,480) # ubah tinggi video
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    revT, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceCascade.detectMultiScale(abuAbu, 1.3, 5) #frame ,scalefactor, minNeighboar
    
    for(x, y , w, h) in wajah:
        cv2.rectangle(frame,(x,y), (x+w, y+h), ( 0, 255, 0), 3)
        
    cv2.imshow('Camera', frame)
    #cv2.imshow('Camera 2', abuAbu)
    k = cv2.waitKey(1) & 0xff
    if k == 27 or k == ord('q'): 
        break

cam.release()
cv2.destroyAllWindows()
    
    