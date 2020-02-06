import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img.jpg', 0)
surf = cv2.xfeatures2d.SURF_create(50000) #create the SURF object

kp, des = surf.detectAndCompute(img, None) #find keypoints and descriptors

final_img = cv2.drawKeypoints(img,kp,None,(255,0,0),4) # draws keypoints to image
plt.imshow(final_img), plt.show()
