#얼굴 인식(판별)
import cv2
import sys, os
from uuid import uuid4

import dlib
#import datetime 
#import time 
import math

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> loze: id=1,  etc
# 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.
#names = ['None', 'JUNHEE', 'Jinsun', 'hyeokjin', 'ksw']
names = ['None', 'JUNHEE', 'Jinsun', 'hyeokjin', 'ksw', 'kyumin'] #id = 5로 test(규민)
#firebase_names = ['None', '준승', 'Jinsun', 'hyeokjin', 'ksw']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

count = 0
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 3,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            #firebase_id = firebase_names[id]
            id = names[id]
            
            #confidence = "  {0}%".format(round(100 - confidence))
            #5초 이상 지속되면 캡쳐해서 파일명을 id(name)으로 저장하고 그 이미지  Firebase로
            #now = datetime.datetime.now()
            #nowDate = now.strftime('%Y-%m-%d')

            #===================================================================================== dlib 추가
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()

            dets = detector(img, 1)
            for k, d in enumerate(dets):
                shape = predictor(img, d)
                color_f = (0, 0, 255) #face - 빨강
                #color_l = (255, 0, 0) #파랑
                line_width  =3
                circle_r  = 3
                fontType = cv2.FONT_HERSHEY_SIMPLEX
                fontSize = 2

                cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), color_f, line_width) #얼굴에 사각형을 그린다
    
                num_of_points_out = 17
                num_of_points_in = shape.num_parts - num_of_points_out
                gx_out = 0
                gy_out = 0
                gx_in = 0
                gy_in = 0

                for i in range(shape.num_parts):
                    shape_point = shape.part(i)
                    if i < num_of_points_out:
                        gx_out = gx_out + shape_point.x / num_of_points_out
                        gy_out = gy_out + shape_point.y / num_of_points_out
                    else:
                        gx_in = gx_in + shape_point.x / num_of_points_in
                        gy_in = gy_in + shape_point.y / num_of_points_in
                theta = math.asin(2*(gx_in - gx_out) / (d.right() - d.left())) 
                radian = theta * 180 / math.pi
                print('얼굴방향: {0: .3f} (각도: {1: .3f}도)'.format(theta, radian))

                if radian < 0:
                   textPrefix = 'left'
                else:
                    textPrefix = 'right'
                textShow = textPrefix + str(round(abs(radian), 1)) + "deg."
                cv2.putText(img, textShow, (d.left(), d.top()), fontType, fontSize, color_f, line_width)


            #============================================================================================

            count+=1
        else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        # cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

