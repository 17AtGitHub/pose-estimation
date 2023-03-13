import cv2
import mediapipe as mp
import PoseModule as pm
import time 

# init the PoseModule lib as an object
pose_det = pm.poseDetector()

cap = cv2.VideoCapture('poseVideos/run.mp4')
pTime = 0

while True:
    # read in image from the captured stream
    success, img = cap.read()
    img = cv2.resize(img, (1080,720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    
    img = pose_det.findPose(img, draw=False)
    lmList = pose_det.findPosition(img, draw=False)
    
    # we are interested to track the landmark at index 14: elbow
    idx = 14
    if len(lmList) != 0:
        print(lmList[idx])
        x = lmList[idx][1]
        y = lmList[idx][2]
        cv2.circle(img, (x,y), 7, (0,255,0), cv2.FILLED)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (100,100), cv2.FONT_HERSHEY_TRIPLEX, 3, (0,255,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    
