import cv2
import numpy as np
import os
import HandTracker as hm

brushThickness = 25
eraserThickness = 100

folder = "Tools"
myList = os.listdir(folder)
print(myList)
overlayList = []

for tool in myList:
    img = cv2.imread(f'{folder}/{tool}')
    overlayList.append(img)

print(len(overlayList))

toolbar = overlayList[9]
drawColor = (255,0,255)

width, height = 1280, 720
toolbar_width = 230

cam = cv2.VideoCapture(0)
cam.set(3,width)
cam.set(4,height)

detector = hm.handDetector(detectionCon=int(0.85))

xp,yp = 0,0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img1 = cam.read()
    img1 = cv2.flip(img1,1)

    # Find Hand Landmarks
    img1 = detector.findHands(img1)
    lmList = detector.findPosition(img1,draw=False)

    if len(lmList)!= 0:
        #print(lmList)


        #tip of index and middle finger
        x1,y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()


        if fingers[1] and fingers[2]:
            cv2.rectangle(img1,(x1,y1-25), (x2 ,y2 + 25), drawColor,cv2.FILLED)
            xp, yp = 0, 0

            if 1050 < x1 < 1280:
                row1_y_min = 0
                row1_y_max = 112
                row2_y_min = 112
                row2_y_max = 223
                row3_y_min = 223
                row3_y_max = 335
                row4_y_min = 335
                row4_y_max = 447

                col1_x_min = 1050
                col1_x_max = 1165
                col2_x_min = 1165
                col2_x_max = 1280

                if row1_y_min < y1 < row1_y_max and col1_x_min < x1 < col1_x_max:
                    toolbar = overlayList[0]
                    drawColor = (255, 113, 82)
                elif row1_y_min < y1 < row1_y_max and col2_x_min < x1 < col2_x_max:
                    toolbar = overlayList[2]
                    drawColor =  (173, 74, 0)
                elif row2_y_min < y1 < row2_y_max and col1_x_min < x1 < col1_x_max:
                    toolbar = overlayList[3]
                    drawColor = (223, 192, 12)
                elif row2_y_min < y1 < row2_y_max and col2_x_min < x1 < col2_x_max:
                    toolbar = overlayList[4]
                    drawColor =  (230, 108, 203)
                elif row3_y_min < y1 < row3_y_max and col1_x_min < x1 < col1_x_max:
                    toolbar = overlayList[5]
                    drawColor = (87, 217, 158)
                elif row3_y_min < y1 < row3_y_max and col2_x_min < x1 < col2_x_max:
                    toolbar = overlayList[6]
                    drawColor =  (89, 222, 255)
                elif row4_y_min < y1 < row4_y_max and col1_x_min < x1 < col1_x_max:
                    toolbar = overlayList[7]
                    drawColor =  (77, 145, 255)
                elif row4_y_min < y1 < row4_y_max and col2_x_min < x1 < col2_x_max:
                    toolbar = overlayList[8]
                    drawColor = (49, 49, 255)
                elif y1 > 500:
                    toolbar = overlayList[1]
                    drawColor = (0 ,0, 0)  # Eraser

                cv2.rectangle(img1, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


        if fingers[1] and fingers[2]==False:
            cv2.circle(img1, (x1,y1),15,drawColor,cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1


            if drawColor == (0, 0, 0):
                 cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                 cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                 cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                 cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

           #Clear Canvas when all fingers are up
        if all (x >= 1 for x in fingers):
             imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img1 = cv2.bitwise_and(img1, imgInv)
    img1 = cv2.bitwise_or(img1, imgCanvas)


    #Placing toolbar
    img1[0:720, width - toolbar_width:] = toolbar
    cv2.imshow("Image",img1)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)



