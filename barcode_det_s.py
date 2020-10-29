import numpy as np
import cv2
import imutils
import copy
import pytesseract
import time


t = time.process_time()



'''
img = cv2.imread("bc.jpg", -1)
#img = imutils.resize(img, width=1800)
# Laterally invert the image / flip the image
# converting from BGR to HSV color space
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

 

# Range for lower red
lower_red = np.array([35,21,204])
upper_red = np.array([45,31,244])
mask1 = cv2.inRange(hsv, lower_red, upper_red)
#cv2.imshow("mask0", mask1)
 

# Range for upper range
lower_red = np.array([109,34,235])
upper_red = np.array([119,44,275])
mask2 = cv2.inRange(hsv,lower_red,upper_red)
#cv2.imshow("mask2", mask2)
# Generating the final mask to detect red color
mask1 = mask1+mask2
#cv2.imshow("mask1", mask1)

cv2.waitKey(0)
'''

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\Tesseract.exe'
file = open("recognized.txt", "w+") 
file.write("") 
file.close() 

image = cv2.imread('photo/90.bmp')
#image = imutils.resize(image, width=1900)
blank_mask = np.zeros(image.shape, dtype=np.uint8)
result = image.copy()
image_copy = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow('image', image)



    # [[75,  44, 252], [85,  54, 292]],
    # [[67,  45, 213], [77,  55, 253]],
    # [[68,  45, 216], [78,  55, 256]],
    # [[79,  40, 255], [89,  50, 295]],
    # [[65,  48, 232], [75,  58, 272]],
    # [[68,  48, 211], [78,  58, 251]],
    # [[65,  48, 209], [75,  58, 249]],
    # [[68,  49, 210], [78,  59, 250]]


    # [[65,48,232],[55,38,192],[75,58,272]],
    # [[62,45,225],[52,35,185],[72,55,265]],
    # [[67,43,212],[57,33,172],[77,53,252]],
    # [[65,45,205],[55,35,165],[75,55,245]],
    # [[65,44,213],[55,34,173],[75,54,253]],
    # [[67,45,211],[57,35,171],[77,55,251]],
    # [[73,42,205],[63,32,165],[83,52,245]],
    # [[67,47,210],[57,37,170],[77,57,250]],
    # [[69,44,215],[59,34,175],[79,54,255]],
    # [[73,45,214],[63,35,174],[83,55,254]]

hsv_bounds = [

    [[55,38,192],[65,48,232]],
    [[52,35,185],[62,45,225]],
    [[57,33,172],[67,43,212]],
    [[55,35,165],[65,45,205]],
    [[55,34,173],[65,44,213]],
    [[57,35,171],[67,45,211]],
    [[63,32,165],[73,42,205]],
    [[57,37,170],[67,47,210]],
    [[59,34,175],[69,44,215]],
    [[63,35,174],[83,55,254]]
]
mask = None
for i in hsv_bounds:
    lower = np.array(i[0])
    upper = np.array(i[1])
    #print(i)
    if (np.all(mask) == np.all(None)):
        mask = cv2.inRange(image, lower, upper)
    else:
        mask = cv2.inRange(image, lower, upper)+mask


result = cv2.bitwise_and(result, result, mask=mask)

reduce_size = 3

mask_ = imutils.resize(mask, width=int(mask.shape[1]/reduce_size))
#cv2.imshow('mask', mask_)
result_ = imutils.resize(result, width=int(result.shape[1]/reduce_size))
#cv2.imshow('result', result_)


# # Perform morphological operations
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
# #cv2.imshow('opening', opening)
# close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
# #cv2.imshow('close', close)

kernel = np.ones((5,5),np.uint8)
res = cv2.dilate(mask,kernel,iterations = 1)
res_ = imutils.resize(res, width=int(res.shape[1]/reduce_size))
#cv2.imshow('res', res_)


#contours = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
cnts_max = cv2.contourArea(cnts_sorted[0])
cnts_max_2 = cv2.contourArea(cnts_sorted[1])


#print("len(cnts_sorted)", len(cnts_sorted))
cnts_upd = []
#print("cnts_upd: ")
for i in cnts_sorted:
    i_tmp = cv2.contourArea(i)
    if (cv2.contourArea(i) > cnts_max*0.8):
        cnts_upd.append(i)
        #print(i_tmp)
#cnts = [i for i in cnts_sorted if np.all(i) > np.all(cnts_max)*0.8]
#111235.5

j = 0

bounding_boxes = []

for i in cnts_upd:
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(blank_mask, [i], (255,255,255))


    blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
    blank_mask_ = imutils.resize(blank_mask, width=int(blank_mask.shape[1]/reduce_size))
    #cv2.imshow('blank_mask'+str(j), blank_mask_)
    image_copy_ = imutils.resize(image_copy, width=int(blank_mask_.shape[1]))
    #cv2.imshow('image_copy_'+str(j), image_copy_)
    result = cv2.bitwise_and(image_copy,image_copy,mask=blank_mask)
    # Crop ROI from result
    x,y,w,h = cv2.boundingRect(blank_mask)
    ROI = result[y:y+h, x:x+w]
    result_ = imutils.resize(result, width=int(result.shape[1]/reduce_size))
    #cv2.imshow('result'+str(j), result_)
    ROI_ = imutils.resize(ROI, width=int(ROI.shape[1]/reduce_size))
    #cv2.imshow('ROI'+str(j), ROI_)




    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)

    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    # if (box[0][0] > box[0][1]):
    #     box = sorted(box, key=lambda x: x[0])
    #     for i in range(len(box)):
    #         box[i] = list(box[i])
    # else:
    #     box = sorted(box, key=lambda x: x[1])
    #     for i in range(len(box)):
    #         box[i] = list(box[i])

  

    #print("box"+str(j), box)

    expand_box_coef = 0.13
    # if ( box[0][1] - (int(box[0][1]*0.01)) < box[3][1] < box[0][1] + (int(box[0][1]*0.01))):
    #     #print(box[0][1] - (int(box[0][1]*0.01)),  box[3][1], box[0][1] + (int(box[0][1]*0.01)))
    #     tmp = copy.deepcopy(box[1])
    #     box[1] = copy.deepcopy(box[3])
    #     box[3] = copy.deepcopy(tmp)



    box[0][0] = int(abs(box[0][0]-box[1][0])*expand_box_coef)+box[0][0]
    box[1][0] = -int(abs(box[0][0]-box[1][0])*expand_box_coef)+box[1][0]

    box[2][0] = -int(abs(box[2][0]-box[3][0])*expand_box_coef)+box[2][0]
    #box[2][1] = max(box[2][1],box[3][1])
    box[3][0] = int(abs(box[2][0]-box[3][0])*expand_box_coef)+box[3][0]

    box[0][1] = int(abs(box[0][1]-box[3][1])*expand_box_coef*1.2)+box[0][1]
    box[1][1] = int(abs(box[0][1]-box[2][1])*expand_box_coef*1.2)+box[1][1]

    


    #print(min(box, key=lambda x: x[0])[0],max(box, key=lambda x: x[0])[0], min(box, key=lambda x: x[1])[1], max(box, key=lambda x: x[1])[1])
    barcode_rectangle = ROI[min(box, key=lambda x: x[1])[1]:max(box, key=lambda x: x[1])[1], min(box, key=lambda x: x[0])[0]:max(box, key=lambda x: x[0])[0], ].copy()
    #cv2.imshow("barcode_rectangle"+str(j), barcode_rectangle)

  
    #barcode_rectangle = cv2.threshold(barcode_rectangle, 170, 255, cv2.THRESH_BINARY)[1]

    barcode_rectangle = cv2.cvtColor(barcode_rectangle, cv2.COLOR_BGR2GRAY) 
    #cv2.imshow("barcode_rectangle_grey"+str(j), barcode_rectangle)
   
    # applying different thresholding  
    # techniques on the input image 
    thresh1 = cv2.adaptiveThreshold(barcode_rectangle, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1) 
    
    thresh2 = cv2.adaptiveThreshold(barcode_rectangle, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1) 
    
    # the window showing output images 
    # with the corresponding thresholding  
    # techniques applied to the input image 
    #cv2.imshow('Adaptive Mean'+str(j), thresh1) 
    #cv2.imshow('Adaptive Gaussian'+str(j), thresh2) 
    
    thresh2_invert = cv2.bitwise_not(thresh2)
    
    # Preprocessing the image starts 
  
    # Convert the image to gray scale 
    #gray = cv2.cvtColor(barcode_rectangle, cv2.COLOR_BGR2GRAY) 
    
    # Performing OTSU threshold 
    #ret, thresh1 = cv2.threshold(thresh2_invert, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    ##cv2.imshow("thresh1"+str(j), thresh1)
    
    # Specify structure shape and kernel size.  
    # Kernel size increases or decreases the area  
    # of the rectangle to be detected. 
    # A smaller value like (10, 10) will detect  
    # each word instead of a sentence. 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    
    # Appplying dilation on the threshold image 
    dilation = cv2.dilate(thresh2_invert, rect_kernel, iterations = 1) 
    
    # Finding contours 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                    cv2.CHAIN_APPROX_NONE) 
    
    # Creating a copy of image 
    im2 = barcode_rectangle.copy() 
    
    
    # Looping through the identified contours 
    # Then rectangular part is cropped and passed on 
    # to pytesseract for extracting text from it 
    # Extracted text is then written into the text file 
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        
        # Drawing a rectangle on copied image 
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        
        # Cropping the text block for giving input to OCR 
        cropped = im2[y:y + h, x:x + w] 
        
        # Open the file in append mode 
        file = open("recognized.txt", "a") 
        
        # Apply OCR on the cropped image 
        text = pytesseract.image_to_string(cropped) 
        #print(str(j) + " " + text)
        # Appending the text into file 
        file.write(text) 
        file.write("\n") 
        
        # Close the file 
        file.close 




    # clahe = cv2.createCLAHE()

    # clahe = clahe.apply(barcode_rectangle)

    # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharpen = cv2.filter2D(clahe, -1, sharpen_kernel)

    # thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # #cv2.imshow('clahe', clahe)
    # #cv2.imshow('sharpen', sharpen)
    # #cv2.imshow('thresh', thresh)
    # cv2.waitKey()




    bounding_boxes.append(np.array([
        [box[0][0]+x, box[0][1]+y],
        [box[1][0]+x, box[1][1]+y],
        [box[2][0]+x, box[2][1]+y],
        [box[3][0]+x, box[3][1]+y]
    ]))


    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(ROI, [np.array(box)], -1, (0, 255, 0), 2)
    ROI_ = imutils.resize(ROI, width=int(ROI.shape[1]/(reduce_size-2)))
    
    #cv2.imshow("Image"+str(j), ROI_)

    cv2.drawContours(image_copy, [bounding_boxes[-1]], -1, (0, 255, 0), 2)
    j += 1

image_copy_ = imutils.resize(image_copy, width=int(image_copy.shape[1]/reduce_size))
#cv2.imshow("res", image_copy_)

'''
contours = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

for c in contours:
    area = cv2.contourArea(c)
    cv2.drawContours(image_copy, [c], -1, (0, 255, 0), 1)
    #if area > min_area and area < max_area:
    #        cv2.drawContours(result, [c], -1, (0, 0, 255), 1)

#cv2.imshow('res', res)

#cv2.imshow('image_copy', image_copy)

'''


'''
#convert from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#get the saturation plane - all black/white/gray pixels are zero, and colored pixels are above zero.
s = hsv[:, :, 1]

#apply threshold on s
ret, thresh = cv2.threshold(s, 8, 255, cv2.THRESH_BINARY)

#invert colors, so every dark spots are now white
image = cv2.bitwise_not(thresh)

cv2.imwrite("image.png", image)
'''

cv2.waitKey()



'''
img=cv2.imread("bc.jpg")
img = imutils.resize(img, width=800)
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([34,18,204])
upper_red = np.array([45,31,244])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

#cv2.imshow("mask0", mask0)

# upper mask (170-180)
lower_red = np.array([108,34,235])
upper_red = np.array([119,45,276])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
#cv2.imshow("mask1", mask1)


# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

#cv2.imshow("output_img", output_img)

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0
#cv2.imshow("output_hsv", output_hsv)
cv2.waitKey(0)
'''

print(time.process_time() - t)