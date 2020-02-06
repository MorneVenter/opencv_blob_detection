import cv2

img = cv2.imread('img.jpg', 0)
surf = cv2.xfeatures2d.SURF_create(50000) #create the SURF object

kp, des = surf.detectAndCompute(img, None) #find keypoints and descriptors
