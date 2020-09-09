

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import copy



def patch2colum(image,patch_size):

    h,w = image.shape

    patch_h = h-8
    patch_w = w-8
    count = 0
    patch_nums = (patch_h+1)*(patch_w+1)



    blocks = np.zeros((patch_size*patch_size,patch_nums))
    weight = np.zeros((h,w))

    for i in range(patch_h+1):
        for j in range(patch_w+1):
            patch = image[i:i+patch_size,j:j+patch_size]
            weight[i:i+patch_size,j:j+patch_size] = weight[i:i+patch_size,j:j+patch_size] + np.ones((patch_size,patch_size))
            v = patch.reshape((patch_size*patch_size,))

            count = count + 1


    return blocks,weight




def omperr(DCT,X,errorGoal):

    n,P = X.shape
    n,K = DCT.shape
    E2 = errorGoal**2*n  

    nmaxNumCoef = 32  

    A = np.zeros((K,P)) 

    for k in range(P):
        x = X[:,k].reshape((64,1))
        residual = x
        currResNorm2 = residual.T.dot(residual)
        indx = []
        j = 0

        while (currResNorm2 > E2) and (j < nmaxNumCoef):
            j = j + 1

            proj = DCT.T.dot(residual)
            pos = np.argmax(np.abs(proj))

            indx.append(pos)

            d = DCT[:,indx].reshape((64,-1))
            d_ = np.linalg.pinv(d)  #求伪逆

            a = np.dot(d_,x)    #稀疏系数

            residual = x-d.dot(a)

            currResNorm2 = residual.T.dot(residual)



        if len(indx)>0:

            A[indx,k] = a.flatten()



    return A



def creatDCT(patch_size,K):

    Pn = int(math.sqrt(K))
    DCT = np.zeros((patch_size, Pn))
    for i in range(Pn):
        V = np.cos(np.asarray([j*i*math.pi/Pn for j in range(8)]))
        if i>0:
            V=V-np.mean(V)

        DCT[:,i] = V/np.linalg.norm(V)

    DCT = np.kron(DCT,DCT)
    return DCT




def Denoising2SC_DCT(image,patch_size,K,sigma):
    blocks,weight = patch2colum(image,8)
    DCT = creatDCT(8,256)

    for i in range(0,blocks.shape[1],100):

        jumpsize = np.asarray([i+100,blocks.shape[1]]).min()

        vecofmeans = np.mean(blocks[:,i:jumpsize],axis=0)

        blocks[:,i:jumpsize] = blocks[:,i:jumpsize] - vecofmeans

        coefs = omperr(DCT,blocks[:,i:jumpsize],1.15*sigma)

        blocks[:,i:jumpsize] = DCT.dot(coefs) + vecofmeans




    image_out = np.zeros((image.shape[0],image.shape[1]))

    for h in range(image.shape[0]-patch_size+1):
        for w in range(image.shape[1]-patch_size+1):
            block = blocks[:,h*(image.shape[1]-patch_size+1)+w].reshape((patch_size,patch_size))
            image_out[h:h + patch_size, w:w + patch_size] = image_out[h:h + patch_size, w:w + patch_size] + block


    image_out = (image+0.034*sigma*image_out) / (1+0.034*sigma*weight)

    return image_out

def ShowDCT (DCT, ShowFlag=0):
    a = []
    D = copy.deepcopy(DCT)
        #归一化
    for i in range (D.shape[1]):
        D[:,i] = D[:,i] - np.min(D[:,i])
        if np.max(D[:,i]) != 0:
            D[:,i] = D[:,i]/np.max(D[:,i])
    if ShowFlag == 1:
        P = np.var (D,axis=0)#列求平局值
        for i in range (len (P)):
            maxn = np.argmax(P)
            a.extend([maxn])
            P[maxn] = -P[maxn]
        I = np.zeros((D.shape[0],D.shape[1]))
        print (D)
        for i in range(D.shape[1]):
            I[:,i] = D[:,a[D.shape[1]-i-1]]
        D = I
    L = int(math.sqrt (DCT.shape[1]))
    n = int(math.sqrt (DCT.shape[0]))
    img_DCT = np.zeros((L*n,L*n))

    for i in range(L):
        for j in range(L):
            for k in range (D.shape[0]):
                img_DCT[i*8+int(k%8)][j*8+int(k/8)] = D[k][i*16+j]
    img_DCT = img_DCT - np.min(D)
    img_DCT = img_DCT/np.max(D)
    img_DCT = np.uint8(img_DCT*255)
    print (img_DCT)

    img_DCT1 = np.zeros((L*24,L*24),np.uint8)
    for i in range(L*8):
        for j in range(L*8):
            img_DCT1[i*3:i*3+3,j*3:j*3+3] = img_DCT[i][j]
    cv.imshow ("DCT Image",img_DCT1)
    cv.waitKey(0)



















