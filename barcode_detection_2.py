import numpy as np
import cv2
import imutils

image = cv2.imread("photo/90.bmp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# equalize lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# edge enhancement
edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, 
                         ksize = 3, scale = 1, delta = 0)
cv2.imshow("Edges", edge_enh)

retval = cv2.imwrite("edge_enh.jpg", edge_enh)

# bilateral blur, which keeps edges
blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)

# use simple thresholding. adaptive thresholding might be more robust
(_, thresh) = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholded", thresh)

retval = cv2.imwrite("thresh.jpg", thresh)

# do some morphology to isolate just the barcode blob
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)
cv2.imshow("After morphology", closed)

retval = cv2.imwrite("closed.jpg", closed)

# find contours left in the image
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
print(box)
image = imutils.resize(image, width=int(image.shape[1]/4))
cv2.imshow("found barcode", image)
cv2.waitKey(0)
