import cv2
from tensorflow.keras.models import load_model
import My_MP_Pose_Manager_local
import numpy as np
import glm

def run():
    smaller_size_for_faster_demo = False

    cnn_image_size = 96
    cols = 1280
    rows = 720
    testForward = True
    vid = ""
    if testForward:
        vid = "forwardSample.mp4"
    else:
        vid = "rearSample.mp4"
    vid = "testVid.mp4"
    cap = cv2.VideoCapture(vid)
    poseManager = My_MP_Pose_Manager_local.My_Mp_Pose_Manager(cols, rows)
    #cnn_model = load_model('myModel6.h5')
    cnn_model = load_model('myModel.h5')
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #totalFrames = 1776
    frameCount = 0
    incorrectMP_Pred = 0
    incorrectCNNPred = 0

    count = 0

    while True:
        if frameCount == totalFrames+1:
            break
        _, frame = cap.read()
        count +=1

        # skipping frames to give illusion of faster frame rate.
        if count%2==0 or count%3==0 or count%6==0 or count%7==0 or count%8==0 or count%9==0:
            continue

        if smaller_size_for_faster_demo:
            cols = 854
            rows = 480
            small_size = (cols, rows)
            frame = cv2.resize(frame, small_size)

        poseManager.process(frame)

        if poseManager.segmentationMask is None:
            print("img is none")
            continue

        inRangeFrame = cv2.inRange(poseManager.segmentationMask, 0.5, 1)
        contours, hierarchy = cv2.findContours(inRangeFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # if len(contours) > 1:
        #     print("More than one contour detected.")
        #     #continue
        # if len(contours) == 0:
        #     print("No contours detected")
        #     continue

        def getBiggestContourFromContours(contours):
            biggestArea = 0
            biggestContour = contours[0]
            for i in range(0, len(contours)):
                area = cv2.contourArea(contours[i])
                if area > biggestArea:
                    if (area > 4):
                        biggestContour = contours[i]
                        biggestArea = area
            return biggestContour
        biggestCountour = getBiggestContourFromContours(contours)
        x, y, w, h = cv2.boundingRect(biggestCountour)

        ## Works, but very inefficient use of memory
        # newSquare = np.zeros((cols, cols, 3), np.uint8)
        # cv2.drawContours(newSquare, contours, -1, (255,255,255), -1)
        # inRangeFrame = cv2.inRange(newSquare, (255,255,255), (255,255,255))
        # img = inRangeFrame

        ################ Might be better just to create a colsxcols mat ##############################
        cropBoxLen = rows
        if w > rows:
            newSquare = np.zeros((w, cols, 3), np.uint8)
            cv2.drawContours(newSquare, contours, -1, (255,255,255), -1)
            newSquare = cv2.inRange(newSquare, (254,254,254), (255,255,255))
            cropBoxLen = w
            inRangeFrame = newSquare

        mid = int(x + w/2 + 0.5)
        halfCropBoxLen = int(cropBoxLen / 2.0 + 0.5)
        tl = glm.ivec2(mid - halfCropBoxLen, 0)
        br = glm.ivec2(tl.x + cropBoxLen, cropBoxLen)
        if tl.x <0:
            tl.x = 0
            br.x = cropBoxLen
        elif br.x > cols:
            diff = br.x - cols
            tl.x = tl.x - diff
            br.x = tl.x + cropBoxLen


        img = inRangeFrame[tl[1]:br[1], tl[0]:br[0]]
        shape = img.shape
        if img.shape[0]== 0 or img.shape[1] == 0: #workaround for '!ssize.empty() in function 'cv::resize' error
            if img.shape[0] == 0:
                print("x is a zero.")
            if img.shape[1] == 0:
                print("y is a zero.")
            continue
        if shape[0] != shape[1]:
            print("NOT A SQUARE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

        size = (cnn_image_size, cnn_image_size)
        img = cv2.resize(img, size)

        leftShoulderPt = poseManager.leftShoulderOpenCV
        leftElbowPt = poseManager.leftElbowOpenCV
        leftWristPt = poseManager.leftWristOpenCV
        rightShoulderPt = poseManager.rightShoulderOpenCV
        rightElbowPt = poseManager.rightElbowOpenCV
        rightWristPt = poseManager.rightWristOpenCV
        leftHipPt = poseManager.leftHipOpenCV
        rightHipPt = poseManager.rightHipOpenCV
        imgDraw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        blue = (255,0,0)
        cyan = (255,255,0)
        magenta = (255,0,255)

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

            i = poseManager

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

        cnn_img = np.asarray(imgDraw) / 255.0
        testLiveCNN = True
        if testLiveCNN:
            y_pred = cnn_model.predict(cnn_img.reshape(1, 96, 96, 3))
            pos = (int(cols / 3), int(rows / 1.5))
            if y_pred < 0.01:
                y_pred = 0.01
            msg = ""
            color = (0, 255, 0)
            rearColor = (255, 0, 0)
            if y_pred >= 0.5:
                msg = "Carlos CNN FORWARD"
            else:
                msg = "Carlos CNN REAR"
                color = rearColor
            cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

            msg = ""
            color = (0, 255, 0)
            if poseManager.rightElbow3D.z < poseManager.rightShoulder3D.z:
                msg = "Google MP FORWARD"
            else:
                msg = "Google MP REAR"
                color = rearColor
            pos = (pos[0], int(rows / 2))
            cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

        msg = str(frameCount)+"/"+str(totalFrames)
        pos = (int(cols/2), int(rows/25))
        cv2.putText(frame, msg,pos,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        frameCount +=1

        armColor = (255,255,0)
        jointColor = (0,0,255)
        radius = 10
        thic = 3
        cv2.line(frame, poseManager.rightShoulderOpenCV, poseManager.rightElbowOpenCV, armColor, 3)
        cv2.line(frame, poseManager.rightElbowOpenCV, poseManager.rightWristOpenCV, armColor, 3)
        cv2.circle(frame, poseManager.rightShoulderOpenCV, radius, jointColor, thic)
        cv2.circle(frame, poseManager.rightElbowOpenCV, radius, jointColor, thic)
        cv2.circle(frame, poseManager.rightWristOpenCV, radius, jointColor, thic)

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

        if count < 2:
            print("Playing Video...")