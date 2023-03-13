import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, mod_cmplx=1, sm_lms=True, en_seg=False, sm_seg=True,
                 min_detConf=0.5, min_trackConf=0.5):
        self.mode=mode
        self.mod_cmplx=mod_cmplx
        self.sm_lms = sm_lms
        self.en_seg=en_seg  
        self.sm_seg=sm_seg
        self.min_detConf=min_detConf
        self.min_trackConf=min_trackConf

        # initializations from the mediapipe library
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.mod_cmplx, self.sm_lms, self.en_seg,
                                     self.sm_seg, self.min_detConf, self.min_trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    # this completes the initializations, now it is needed to define the methods in this class
    # for pose detection and landmarks labelling
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # if a human pose is detected, print the points as landmarks
        if(self.results.pose_landmarks):
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                               self.mpPose.POSE_CONNECTIONS)

        return img
    #
    def findPosition(self, img, draw=True):
        self.lmList = []
        if(self.results.pose_landmarks):
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # calculation of the angles
        angle = 180-int(math.degrees(math.atan2(y2-y3,x2-x3)-math.atan2(y1-y2,x1-x2)))
        if angle < 0:
            angle += 360
        # print(angle)
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.circle(img, (x1,y1), 7, (0,0,255), -1)
            cv2.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2,y2), 7, (0,0,255), -1)
            cv2.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3,y3), 7, (0,0,255), -1)
            cv2.circle(img, (x3, y3), 15, (0,0,255), 2)
            cv2.putText(img, str(angle), (x2-50,y2+50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
        return angle

def main():
    cap = cv2.VideoCapture('poseVideos/practice.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1080, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = detector.findPose(img,draw=True)
        lmList = detector.findPosition(img, draw=False)

        mark = 14
        # print(lmList)
        if len(lmList) != 0:
            print(lmList[mark])

        cv2.circle(img, (lmList[mark][1],lmList[mark][2]), 10, (255,0,0), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (100,100), cv2.FONT_HERSHEY_TRIPLEX, 3, (255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



# if we are running only this file, then it will call the main function
# but if we are using it as a module, it will not run the main function
if __name__ == "__main__":
    main()
