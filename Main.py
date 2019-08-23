import numpy as np
import cv2
import faceDetector as fd
from playsound import playsound

cap = cv2.VideoCapture(0)


while(True):
    ret, frame = cap.read()
    facecount, frame = fd.faceDetection(frame)
    print(facecount)
    print("Faces in scene ",len(facecount))

    for(x,y,w,h) in facecount:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),thickness=2)

    cv2.imshow("Facetrack",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    


cap.release()
cv2.destroyAllWindows()