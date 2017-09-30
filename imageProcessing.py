import cv2
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
from matplotlib import pyplot as plt
import numpy as np
from cv2 import drawContours
from operator import itemgetter
import math
import csv
import path

name = ""
answer_key = {0:2, 1:3, 2:1, 3:1, 4:4, 5:1, 6:3, 7:3, 8:1, 9:3, 10:1, 11:2, 12:3, 13:3, 14:2, 
              15:1, 16:4, 17:2, 18:3, 19:2, 20:4, 21:3, 22:4, 23:2, 24:4, 25:3, 26:4, 27:4,
              28:2, 29:3, 30:2, 31:2, 32:4, 33:3, 34:2, 35:3, 36:2, 37:3, 38:3, 39:1, 40:2, 41:2,
              42:3, 43:3, 44:2}
smallCircles = 0
circles = 0 # the 2 big circles in the bottom
miny = 0   #global variable corresponding to min value of y in clipped image
maxy = 0   #global variable corresponding to max value of y in clipped image
minx = 0   #global variable corresponding to min value of x in clipped image
maxx = 0   #global variable corresponding to max value of x in clipped image
dify = 40
difx = 40


def readImage(path):
    global name
    name = path
    img = cv2.imread(path)
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return grayImg

def adjust_orientation(img):
    global circles
    circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1.2, 600,
              param1=100,
              param2=30,
              minRadius=30,
              maxRadius=40)

    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=itemgetter(1))[-2:]
    plt.figure(1)
    plt.imshow(img, cmap='gray')
    #circles = sorted(circles, key=itemgetter(0))[-2:]
    # rotating the image
    deltax = circles[1][0] - circles[0][0]
    deltay = circles[1][1] - circles[0][1]
    
    angle = math.degrees(math.atan(float(deltay)/deltax))
    # rotated image
    rotated = imutils.rotate(img, angle)
    return rotated

def get_answer_rectangle(img):
    global miny, maxy
    global circles, smallCircles
    img = img[730:circles[0][1]-100, :]
    img = img[:, 125:-125]
    smallCircles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1.2, 10,
              param1=100,
              param2=25,
              minRadius=10,
              maxRadius=20)
    if smallCircles is not None:
        smallCircles = sorted(smallCircles[0], key=itemgetter(1))
        smallCircles = smallCircles[1:-1]
    miny = smallCircles[0][1]
    maxy = smallCircles[-1][1]
    width = np.shape(img)[1]
    img1 = img[:, 0:width/3]
    img2 = img[:, width/3:2*width/3]
    img3 = img[:, 2*width/3:width]
    return [img1, img2, img3]

def thresholdedImg(imgs):
    for i in range(3):
        imgs[i] = cv2.threshold(imgs[i], 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
    return imgs

def detectBubbles(img):
    global miny
    smallCircles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1.2, 10,
              param1=100,
              param2=35,
              minRadius=10,
              maxRadius=20)
    if smallCircles is not None:
        smallCircles = sorted(smallCircles[0], key=itemgetter(1))
    return smallCircles

def detectQuestions(ts):
    QB1 = detectBubbles(ts[0])[1:]
    QB2 = detectBubbles(ts[1])
    QB3 = detectBubbles(ts[2])[:-1]
    return [QB1,QB2,QB3]


def gradeV2(imgs):
    global answer_key
    global minx, maxx, difx
    global miny, maxy, dify
    score = 0
    Questions = detectQuestions(imgs)
    thr = thresholdedImg(imgs)
    
    for i in range(3):
        Questions[i] = sorted(Questions[i], key=itemgetter(0))
        minx = Questions[i][0][0]
        maxx = Questions[i][-1][0]
        Questions[i] = sorted(Questions[i],
            key=itemgetter(1))
        myCoordinates = []
        count = 0
        previousY = -1
        for q in Questions[i]:
            (x,y,r) = q
            if y >= previousY-8 and y <= previousY+8:
                # i've commented this out cuz i need just 1 choice at each line
                myCoordinates[count-1].append((x,y))
            else :
                myCoordinates.append([(x,y)])
                count += 1
            previousY = y
        # if Questions areas are not 15 in a the list
        if len(myCoordinates) < 15:
            prevY = -1
            ind = []
            for k in myCoordinates:
                y =  k[0][1]
                if y > prevY + dify - 10 and y < prevY + dify + 10:
                    pass
                elif myCoordinates.index(k) == 0:
                    pass
                else:
                    index = myCoordinates.index(k)
                    val = y - prevY
                    while val > 50:
                        ind.append(prevY+dify)
                        val -= dify
                        index += 1
                        prevY = prevY+dify
                prevY = y
            for l in ind:
                myCoordinates.append([(minx, int(l))])
            myCoordinates = sorted(myCoordinates, key= lambda tup:tup[0][1])
            while len(myCoordinates) < 15:
                if myCoordinates[0][0][1] - dify > miny-15 :
                    myCoordinates.insert(0, [(minx, int(myCoordinates[0][0][1]-dify))])
                else:
                    myCoordinates.append([(minx,int(myCoordinates[-1][0][1]+dify))])
        # computing results
        xcoord = 0
        ycoord = 0
        for j in myCoordinates:
            xcoord = int(minx)
            ycoord = int(j[0][1])
            bubbled = None
            duplicatedChoice = False
            for o in range(4):
                temp = thr[i][ycoord-15:ycoord+15, xcoord-20:xcoord+20]
                total = cv2.countNonZero(temp)
                xcoord += int(difx)
                if bubbled != None and bubbled[0] > 150 and total > 150 and (float(bubbled[0])/total>1.3 or float(total)/bubbled[0]>1.3):
                    pass
                elif bubbled != None and bubbled[0] > 150 and total > 150:
                    duplicatedChoice = True
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, o)
            ans = answer_key[myCoordinates.index(j)+ i * 15]
            if ans == bubbled[1]+1 and bubbled[0] > 150 and not duplicatedChoice:
                score += 1
    return score
        

def grade(img):
    img = adjust_orientation(img)
    imgs = get_answer_rectangle(img)
    #imgs = strengthenCircles(imgs)
    # get tresholded images
    return gradeV2(imgs)
    

readerFile = open("Sample Submission.csv", 'r')
writerFile = open("my Submission4.csv", 'w')
writer = csv.writer(writerFile)
writer.writerow(['FileName', 'Mark'])
reader = csv.reader(readerFile)
next(reader, None)
for line in reader:
    try:
        path = "test/"+line[0]
        image = readImage(path)
        grd = grade(image)
        data = [line[0],grd]
        writer.writerow(data)
    except:
        print name
        exit()
writerFile.close()
readerFile.close()

#plt.imshow(image, cmap="gray")
#plt.show()

'''
    cv2.drawContours(ts[0], QB1, -1, 255, 15)
    cv2.drawContours(ts[1], QB2, -1, 255, 15)
    cv2.drawContours(ts[2], QB3, -1, 255, 15)
    plt.figure(0)
    plt.imshow(ts[0], cmap="gray")
    plt.figure(1)
    plt.imshow(ts[1], cmap="gray")
    plt.figure(2)
    plt.imshow(ts[2], cmap="gray")
    plt.show()
       print ans, bubbled[1]+1, score
        print len(myCoordinates)
        for iiiii in myCoordinates:
            print iiiii
        print miny
        print minx, (maxx-minx+20)/4
        plt.imshow(thr[i], cmap='gray')
        plt.show()
        
      

'''


'''
bubble 1 : from 110 to 140
bubble 2 : from 150 to 180
bubble 3 : from 190 to 220
bubble 4 from 220 to the end
'''





