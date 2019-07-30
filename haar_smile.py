# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 22:09:28 2019

@author: ujjwal.anand
"""
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

vc = cv2.VideoCapture(0)
while(True):
    _,frame = vc.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    face = face_cascade.detectMultiScale(grey,1.8,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        areaOfInterest = frame[y:y+h,x:x+w]
        smiles = smile_cascade.detectMultiScale(areaOfInterest,1.3,2)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(areaOfInterest,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
        
    cv2.imshow("",frame)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break;
vc.release()
cv2.destroyAllWindows()