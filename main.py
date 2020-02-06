import cv2
import matplotlib.pyplot as plt

for i in range(4):
    print('####################################################################')
    print('IMAGE ' + str(i+1))
    print('####################################################################')
    img = cv2.imread('spot_'+str(i+1)+'.jpg', 0)
    surf = cv2.xfeatures2d.SURF_create(7500) #create the SURF object

    if not(surf.getUpright()):
        surf.setUpright(True)
    if not(surf.getExtended()):
        surf.setExtended(True)

    kp, des = surf.detectAndCompute(img, None) #find keypoints and descriptors
    print('Size of descriptors: ' + str(surf.descriptorSize()))
    print('Number of keypoints: ' + str(len(kp)))
    final_img = cv2.drawKeypoints(img,kp,None,(255,0,0),4) # draws keypoints to image
    for point in kp:
        print('Sunspot at:' + str(round(point.pt[0],2)) +' ' + str(round(point.pt[1],2)))
    plt.imshow(final_img), plt.show()
    print('####################################################################')
