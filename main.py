import MP_vs_My_CNN
import cv2
import glm # NOTE: Use 'pip install PyGLM', DO NOT USE 'pip install glm'
import pickle
import numpy as np
import My_CNN_Carlos

# If skipTraining is True, a pre-trained model, myModel.h5, will be used. A new model will not be trained if skipTraining is True.
skipTraining = False

if not skipTraining:
    class Frames_Collection:
        def __init__(self, framesNativeCols, framesNativeRows, cropTL, cropBR, cameraName, focalLength_mm, OAK_D_Frames):
            self.framesNativeCols = framesNativeCols
            self.framesNativeRows = framesNativeRows
            self.cameraName = cameraName
            self.focalLength_mm = focalLength_mm
            self.OAK_D_Frames = OAK_D_Frames
            self.cropTL = cropTL
            self.cropBR = cropBR

    class OAK_Frame:

        def __init__(self, img, cnn_image_size, topLeft, left_shoulderPt, leftElbowPt, leftWristPt, right_shoulderPt, rightElbowPt, rightWristPt, leftHipPt, rightHipPt, mp_results_pose_landmarks, poseManager):
            cropBoxLen = img.shape[0]

            size = (cnn_image_size, cnn_image_size)
            img = cv2.resize(img, size)

            #imgForShow = cv2.resize(img, (img.shape[0] * 4, img.shape[1] * 4))
            #cv2.imshow("img", imgForShow)

            self.allPts = [left_shoulderPt, leftElbowPt, leftWristPt, right_shoulderPt, rightElbowPt, rightWristPt,
                           leftHipPt, rightHipPt]
            self.allPts = tuple(self.allPts) #to prevent accidental modification of pts
            self.allPts3D = []

            tl =topLeft


            scalePts = True
            if scalePts:
                for pt in self.allPts:
                    #adjust size for square roi
                    ptx = pt.x - tl.x

                    # adjust for scale
                    ptx = (ptx * cnn_image_size)/cropBoxLen
                    pty = (pt.y * cnn_image_size)/cropBoxLen
                    ptz = (pt.z * cnn_image_size)/cropBoxLen

                    self.allPts3D.append(glm.vec3(ptx, pty, ptz))

            self.frame = img

            self.allPtsOpenCV = []
            for pt in self.allPts3D:
                ptCV = glm.ivec2(pt[0], pt[1])
                ptCV = tuple(ptCV)
                self.allPtsOpenCV.append(ptCV)

            # draw = False
            # if draw:
            #     ptA = self.allPtsOpenCV[3]
            #     ptB = self.allPtsOpenCV[4]
            #     ptC = self.allPtsOpenCV[5]
            #     imgDraw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #     cv2.line(imgDraw, ptA, ptB, (0, 255, 0), 1)
            #     cv2.line(imgDraw, ptB, ptC, (0,0,255),1)
            #     showSize = (int(imgDraw.shape[0]*5), int(imgDraw.shape[1]*5))
            #     imgForShow = cv2.resize(imgDraw, showSize)
            #     cv2.imshow("imgCrop", imgForShow)

            # self.left_shoulderPt = left_shoulderPt
            # self.left_elbowPt = leftElbowPt
            # self.leftWristPt = leftWristPt
            # self.right_shoulderPt = right_shoulderPt
            # self.rightElbowPt = rightElbowPt
            # self.rightWristPt = rightWristPt
            # self.leftHipPt = leftHipPt
            # self.rightHipPt = rightHipPt
            self.mp_results_pose_landmarks = mp_results_pose_landmarks
            #self.mp_results = mp_results

            #for id, lm in enumerate(mp_results.pose_landmarks.landmark):
                #print("id", id)
            #self.rightShoulderVisibility = mp_results.pose_landmarks.landmark[12].visibility
            self.rightShoulderVisibility = poseManager.rightShoulderVisibility
            self.rightElbowVisibility = poseManager.rightElbowVisibility
            self.rightWristVisibility = poseManager.rightWristVisibility

            self.leftShoulderVisibility = poseManager.leftShoulderVisibility
            self.leftHipVisibility = poseManager.leftHipVisibility
            self.rightHipVisibility = poseManager.rightHipVisibility


    datasetFiles = ['oakDframeSampleRear4.pickle', 'oakDframeSampleForward4.pickle', 'oakDframeSampleRear5.pickle',
                    'oakDframeSampleForward5.pickle',
                    'oakDframeSampleRear6.pickle', 'oakDframeSampleForward6.pickle']
    rearDataSet = []
    forwardDataSet = []

    for dataIdx, dataVal in enumerate(datasetFiles):
        # with open(i, 'rb') as f:
        with open(dataVal, 'rb') as f:
            framesCollection = pickle.load(f)
            oakdframes = framesCollection.OAK_D_Frames
            print("len", len(oakdframes))
            count = 0
            # stopAtFrame = len(oakdframes)
            stopAtFrame = 2000

            validOAK_D_frames = []

            for idx, i in enumerate(oakdframes):
                blue = (255, 0, 0)
                magenta = (255, 0, 255)
                cyan = (255, 255, 0)
                red = (0, 0, 255)
                green = (0, 255, 0)
                mp_results_pose_landmarks = i.mp_results_pose_landmarks
                if idx == stopAtFrame:
                    break

                draw = True
                if draw:
                    # MP
                    # 12 right shoulder
                    # self.mp_results_pose_landmarks = results.pose_landmarks
                    # for id, lm in enumerate(results.pose_landmarks.landmark):
                    # print(mp_results_pose_landmarks)
                    # for id, lm in enumerate(mp_results_pose_landmarks.landmark):
                    # print("id", id)
                    # print(mp_results_pose_landmarks)
                    # print("-------------------------------")

                    # allPts = [
                    #           0   left_shoulderPt,
                    #           1   leftElbowPt,
                    #           2   leftWristPt,
                    #           3   right_shoulderPt,
                    #           4   rightElbowPt,
                    #           5   rightWristPt,
                    #           6   leftHipPt,
                    #           7   rightHipPt
                    #           ]
                    leftShoulderPt = i.allPtsOpenCV[0]
                    leftElbowPt = i.allPtsOpenCV[1]
                    leftWristPt = i.allPtsOpenCV[2]
                    rightShoulderPt = i.allPtsOpenCV[3]
                    rightElbowPt = i.allPtsOpenCV[4]
                    rightWristPt = i.allPtsOpenCV[5]
                    leftHipPt = i.allPtsOpenCV[6]
                    rightHipPt = i.allPtsOpenCV[7]
                    imgDraw = cv2.cvtColor(i.frame, cv2.COLOR_GRAY2BGR)

                    # torso
                    cv2.line(imgDraw, rightShoulderPt, leftShoulderPt, blue, 1)
                    cv2.line(imgDraw, rightShoulderPt, rightHipPt, blue, 1)
                    cv2.line(imgDraw, rightHipPt, leftHipPt, blue, 1)
                    cv2.line(imgDraw, leftHipPt, leftShoulderPt, blue, 1)

                    drawVisibility = True
                    if drawVisibility:
                        cv2.line(imgDraw, rightShoulderPt, rightElbowPt, blue, 1)
                        cv2.line(imgDraw, rightElbowPt, rightWristPt, blue, 1)
                        cv2.line(imgDraw, leftShoulderPt, leftElbowPt, blue, 1)
                        cv2.line(imgDraw, leftElbowPt, leftWristPt, blue, 1)

                        blue = np.asarray(blue)
                        blueVis = blue * i.rightShoulderVisibility
                        blueVis = glm.ivec3(blueVis)
                        if blueVis[0] == 0:
                            blueVis[0] = 1
                        if blueVis[0] < 254:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        if blueVis[0] == 255:
                            blueVis[0] = 244
                        cv2.line(imgDraw, rightShoulderPt, rightShoulderPt, blueVis, 1)

                        cyan = np.asarray(cyan)
                        cyanVis = cyan * i.rightElbowVisibility
                        cyanVis = glm.ivec3(cyanVis)
                        if cyanVis[0] == 0:
                            cyanVis[0] = 1
                        if cyanVis[1] == 0:
                            cyanVis[1] = 1
                        cv2.line(imgDraw, rightElbowPt, rightElbowPt, cyanVis, 1)

                        magenta = np.asarray(magenta)
                        magentaVis = magenta * i.rightWristVisibility
                        magentaVis = glm.ivec3(magentaVis)
                        if magentaVis[0] == 0:
                            magentaVis[0] = 1
                        if magentaVis[2] == 0:
                            magentaVis[2] = 1
                        cv2.line(imgDraw, rightWristPt, rightWristPt, magentaVis, 1)


                    else:
                        # right arm
                        cv2.line(imgDraw, rightShoulderPt, rightElbowPt, magenta, 1)
                        cv2.line(imgDraw, rightElbowPt, rightWristPt, cyan, 1)

                    # left arm
                    # cv2.line(imgDraw, leftShoulderPt, leftElbowPt, cyan, 1)
                    # cv2.line(imgDraw, leftElbowPt, leftWristPt, cyan, 1)

                    # if dataIdx == 0 or dataIdx == 2 or dataIdx == 4:
                    if dataIdx % 2 == 0:
                        rearDataSet.append(imgDraw)
                    else:
                        forwardDataSet.append(imgDraw)

                    showImshow = False
                    if dataIdx == 4 or dataIdx == 5:
                        if showImshow:
                            showSize = (int(imgDraw.shape[0] * 5), int(imgDraw.shape[1] * 5))
                            imgForShow = cv2.resize(imgDraw, showSize, interpolation=cv2.INTER_NEAREST_EXACT)
                            org = (int(imgForShow.shape[0] / 2), int(imgForShow.shape[1] / 13))
                            cv2.putText(imgForShow, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow(str(dataIdx), imgForShow)
                            cv2.waitKey(1)

    useImshow = False
    if useImshow:
        for i in rearDataSet:
            cv2.imshow("rear", i)
            cv2.waitKey(1)
        for i in forwardDataSet:
            cv2.imshow("forward", i)
            cv2.waitKey(1)
    build_cnn = True
    if build_cnn:
        My_CNN_Carlos.My_CNN_Carlos(rearDataSet, forwardDataSet)

MP_vs_My_CNN.run()