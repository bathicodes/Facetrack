import cv2
import numpy as np

def faceDetection(frame):
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceCount = face_haar_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5)

    return faceCount, frame