import cv2
import mediapipe as mp
import PoseModule as pm
import time
import numpy as np

cap = cv2.VideoCapture('trainer/curls2.mp4')
pTime=0
pose_det = pm.poseDetector()
count=0
dir=0

while True:
    success, img = cap.read()
    # image resizing
    scaleFactor=30
    nw = int(img.shape[1]*scaleFactor/100)
    nh = int(img.shape[0]*scaleFactor/100)
    dim=(nw,nh)
    img=cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # img=cv2.imread('trainer/test.png')
    img=pose_det.findPose(img, False)
    lmList=pose_det.findPosition(img,False)

    if len(lmList) != 0:
        # right hand
        # pose_det.findAngle(img, 12,14,16)
        # #left hand
        angle = pose_det.findAngle(img, 11,13,15)

        # down completion at 160
        # up completion at 55
        # map the rnage from 0 to 100 range
        per = np.interp(angle, (55,160), (100,0))
        color=(255,0,255)
        if per == 100:
            color=(0,255,0)
            if dir == 0:
                count+=0.5
                dir=1
        if per == 0:
            color=(0,255,0)
            if dir == 1:
                count += 0.5
                dir=0

        h,w = img.shape[1], img.shape[0]
        print(h, w)
        # box for containing the count
        cv2.rectangle(img, (0,h-50), (200,h+200), (0,255,0), -1)
        cv2.putText(img, str(count), (50, h+70), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,0,255), 2)

        # box for displaying the levels
        cv2.rectangle(img, (100,150), (200,550),color,3)
        height = int(np.interp(per, (0,100), (550,150),-1))
        cv2.rectangle(img, (100,height), (200,550),color,-1)
        cv2.putText(img, f'{str(int(per))}%', (110,120), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color, 2)
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)