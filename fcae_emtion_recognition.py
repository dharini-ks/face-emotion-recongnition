from facial_emotion_recognition import EmotionRecognition
import cv2
import torch
model_path = 'path/to/your/model.pth'

er=EmotionRecognition(device='cpu', model_path=model_path)
cam=cv2.VideoCapture(1)
while True:
    success,frame=cam.read()
    frame=er.recognise_emotion(frame,return_type='BGR')
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()



