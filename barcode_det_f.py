try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
from skimage import filters, io
import numpy as np
import math
import imutils
import time

cv2.namedWindow("out", cv2.WINDOW_NORMAL)

def get_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / max(M["m00"], 1e-6))
    cY = int(M["m01"] / max(M["m00"], 1e-6))

    return cX, cY

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    print("M", M)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box

    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop


def findBars(image=cv2.imread(r'C:\Py\agroocr\img\201023_95_2.bmp')):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    
    gradX = cv2.Scharr(gray, ddepth=ddepth, dx=1, dy=0)  # , ksize=-1)
    
    gradY = cv2.Scharr(gray, ddepth=ddepth, dx=0, dy=1)  # , ksize=-1)
    # gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    # gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    
    gradient = cv2.convertScaleAbs(gradient)
    gradient_ = imutils.resize(gradient, width=int(gradient.shape[1]/3))
    cv2.imshow("gradient_", gradient_)
    # blur and threshold the image
    # blurred = cv2.blur(gradient, (9, 9))
    blurred = cv2.blur(gradient, (14, 14))
    blurred_ = imutils.resize(blurred, width=int(blurred.shape[1]/3))
    cv2.imshow("blurred_", blurred_)
    (_, thresh) = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 4))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed_ = imutils.resize(closed, width=int(closed.shape[1]/3))
    cv2.imshow("closed_", closed_)
    # perform a series of erosions and dilations
    # closed = cv2.erode(closed, None, iterations=4)
    # closed = cv2.dilate(closed, None, iterations=4)
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    large_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:]
    j=0


    res = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for c in cnts[:]:
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        
        w1 = rect[1][0]
        w2 = rect[1][1]
        if 60 < w1 < 70 and 120 < w2 < 140:
            print(rect)
            box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
            
            box = np.int0(box)
            print("box:", box)
            
            # draw a bounding box arounded the detected barcode
            # and display the image

            mask = np.zeros(image.shape[:2],np.uint8)
            print(image.shape[:2])
            print(str(box[1][0]) + " " + str(box[3][0]) + "   " + str(box[2][1]) +" "+str(box[0][1]))
            offset = 0.05
            mask[int(box[2][1]-(box[0][1]-box[2][1])*offset):int(box[0][1]+(box[0][1]-box[2][1])*offset),int(box[1][0]-(box[3][0]-box[1][0])*offset):int(box[3][0]+(box[3][0]-box[1][0])*offset)] = 255
            res = res + cv2.bitwise_and(image,image,mask = mask)
            cv2.imshow("mask"+str(j), mask)
            cv2.drawContours(image, [box], -1, (0, 255, 0), 1)

        j+=1


    barcode_rectangle = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
    thresh2 = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1) 
    thresh2_invert = cv2.bitwise_not(thresh2)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    dilation = cv2.dilate(thresh2_invert, rect_kernel, iterations = 1) 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    cv2.imshow("dilation", dilation)
    file = open("recognized.txt", "a") 
    text = pytesseract.image_to_string(cropped) 
    file.write(text) 
    file.write("\n") 

    # cv2.imwrite('res_grad.jpg', image)
    cv2.imshow("out", image)
    cv2.imshow("res", res)
    cv2.imwrite("res.jpg", res) 
    cv2.imwrite("image.jpg", image) 
    cv2.waitKey(0)


def readSymbols(im):
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # retval, threshold = cv2.threshold(image,170,255,cv2.THRESH_BINARY)

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)

    # Simple image to string
    print(pytesseract.image_to_string(sharpened, config='digits'))
    # lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
    cv2.imshow("", sharpened)
    cv2.waitKey(0)


# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

file_path = r'90.bmp'  # 201023_95_2.bmp'
image = cv2.imread(file_path)
t = time.process_time()
findBars(image)
print(time.process_time() - t)

# to do:
# check findBars on different images
# connect findBars and readSymbols (widen rects from findBars and remove bars from them, send pics with symbols only to readSymbols )

