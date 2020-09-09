import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import denoiseImageDCT


def psnr(image_p,image_o):



    gap = np.asarray(image_p - image_o)
    gap = gap.flatten()
    mn = len(gap)
    print('1', mn)
    print('2',gap)
    s = np.dot(gap.T,gap)
    MSE = s/mn
    print('3',MSE)

    return 20*np.log10(255/np.sqrt(MSE))

sigma = 25
image = cv.imread('barbara.png',cv.IMREAD_GRAYSCALE)

n = np.random.randn(512,512)
image_o = sigma*n+image

image_o = np.asarray(image_o)

image_denoise1 = denoiseImageDCT.Denoising2SC_DCT(image_o,8,256,sigma)



plt.figure()
plt.imshow(image_o,'gray')
plt.title(psnr(image_o,image))
plt.figure()
plt.imshow(image_denoise1,'gray')
plt.title(psnr(image_denoise1,image))

plt.show()










