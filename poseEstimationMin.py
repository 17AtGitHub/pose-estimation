import cv2
import mediapipe as mp
import time

# read in the video file

cap = cv2.VideoCapture('poseVideos/run.mp4')

pTime = 0
cTime = 0

# creating the mode, using classes from mp lib
mpPose = mp.solutions.pose
pose = mpPose.Pose()
# keeping the default parameters in Pose
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if success == True:
        frame = cv2.resize(img, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)
        if (results.pose_landmarks):
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = frame.shape
                cx, cy = (int)(lm.x*w), (int)(lm.y*h)
                print(id, cx, cy)
                cv2.circle(frame, (cx,cy), 7, (255,0,0), -1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Image", frame)

    cv2.waitKey(1)
