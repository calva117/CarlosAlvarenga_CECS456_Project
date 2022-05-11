import enum
import cv2
import mediapipe as mp
import glm
import numpy as np

class MP_Name_ID(enum.Enum): #MP = MediaPipe
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER = 11
    RIGHT_HIP = 24
    LEFT_HIP = 23

    RIGHT_WRIST = 16
    RIGHT_ELBOW = 14

    LEFT_WRIST = 15
    LEFT_ELBOW = 13

    LEFT_KNEE = 25
    LEFT_ANKLE = 27

    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28

class MP_Name_ID(enum.Enum): #MP = MediaPipe
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER = 11
    RIGHT_HIP = 24
    LEFT_HIP = 23

    RIGHT_WRIST = 16
    RIGHT_ELBOW = 14

    LEFT_WRIST = 15
    LEFT_ELBOW = 13

    LEFT_KNEE = 25
    LEFT_ANKLE = 27

    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28



class My_Mp_Pose_Manager:

    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.halfcols = cols/2
        self.halfrows = rows/2
        landmarksTorso = [MP_Name_ID.LEFT_SHOULDER, MP_Name_ID.RIGHT_SHOULDER, MP_Name_ID.RIGHT_HIP, MP_Name_ID.LEFT_HIP]
        self.landmarkTorso_shape_dictionary = {}

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.enableSegmentation = True
        self.pose = self.mpPose.Pose(enable_segmentation=True)

        self.landmark_coord_dictionary = {}
        for id in MP_Name_ID:
            self.landmark_coord_dictionary[id.value] = (0,0)
        self.lines = []
        self.sternumPt = glm.vec3(0,0,0)
        self.lowestYvalue = 0

        self.leftShoulderOpenCV = (0, 0)
        self.rightShoulderOpenCV = (0, 0)
        self.leftHipOpenCV = (0, 0)
        self.rightHipOpenCV = (0, 0)

        self.rightElbowOpenCV = (0,0)
        self.leftElbowOpenCV = (0,0)
        self.leftWristOpenCV = (0,0)
        self.rightWristOpenCV = (0,0)
        self.segmentationMask = np.zeros((cols,rows,3), np.uint8)

        self.mp_results_pose_landmarks = []
        #self.mp_results = []
        
        self.rightShoulderVisibility = 0
        self.rightElbowVisibility = 0
        self.rightWristVisibility = 0
        
        self.leftShoulderVisibility = 0
        self.leftHipVisibility = 0
        self.rightHipVisibility = 0

        self.rightShoulder3D = glm.vec3(0,0,0)
        self.rightElbow3D = glm.vec3(0,0,0)
        self.rightWrist3D = glm.vec3(0,0,0)
        
        self.leftShoulder3D = glm.vec3(0,0,0)
        self.leftHip3D = glm.vec3(0,0,0)
        self.rightHip3D = glm.vec3(0,0,0)

    def process(self, frame, drawSkeleton = False, imshow = False):
        pose = self.pose
        mpDraw = self.mpDraw
        mpPose = self.mpPose

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        lowestYvalue = 999999
        if results.pose_landmarks:
            self.mp_results = results
            self.mp_results_pose_landmarks = results.pose_landmarks
            if drawSkeleton:
                mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
            #for id, lm in enumerate(self.mp_results_pose_landmarks.landmark):
                h, w, c = frame.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                #cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                self.landmark_coord_dictionary[id] = (cx, cy)
                if id == MP_Name_ID.RIGHT_ANKLE.value or id == MP_Name_ID.LEFT_ANKLE.value:
                    if cy < lowestYvalue:
                        lowestYvalue = cy

                if id == MP_Name_ID.LEFT_SHOULDER.value:
                    self.leftShoulderOpenCV = (cx,cy)
                    self.leftShoulderVisibility = lm.visibility
                    self.leftShoulder3D = glm.vec3(cx, cy, lm.z*w)
                elif id == MP_Name_ID.RIGHT_SHOULDER.value:
                    self.rightShoulderOpenCV = (cx,cy)
                    self.rightShoulderVisibility = lm.visibility
                    self.rightShoulder3D = glm.vec3(cx, cy, lm.z*w)
                elif id == MP_Name_ID.LEFT_HIP.value:
                    self.leftHipOpenCV = (cx,cy)
                    self.leftHipVisibility = lm.visibility
                    self.leftHip3D = glm.vec3(cx, cy, lm.z*w)
                elif id == MP_Name_ID.RIGHT_HIP.value:
                    self.rightHipOpenCV = (cx,cy)
                    self.rightHipVisibility = lm.visibility
                    self.rightHip3D = glm.vec3(cx, cy, lm.z*w)
                elif id == MP_Name_ID.RIGHT_ELBOW.value:
                    self.rightElbowOpenCV = (cx,cy)
                    self.rightElbowVisibility = lm.visibility
                    self.rightElbow3D = glm.vec3(cx, cy, lm.z*w)
                elif id == MP_Name_ID.LEFT_ELBOW.value:
                    self.leftElbowOpenCV = (cx,cy)
                elif id == MP_Name_ID.LEFT_WRIST.value:
                    self.leftWristOpenCV = (cx,cy)
                elif id == MP_Name_ID.RIGHT_WRIST.value:
                    self.rightWristOpenCV = (cx,cy)
                    self.rightWristVisibility = lm.visibility
                    self.rightWrist3D = glm.vec3(cx, cy, lm.z*w)


        if self.enableSegmentation:
            if results.pose_landmarks:
                self.segmentationMask = results.segmentation_mask


        if imshow:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)