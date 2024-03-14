import cv2

img1 = cv2.imread('../../data/240130/Cameraman64_.png')/255
img2 = cv2.imread('../../data/240213/Cameraman64_0.05.png')/255

ssim, _ = cv2.quality.QualitySSIM_compute(img1, img2)

print('SSIM:', ssim[0])
