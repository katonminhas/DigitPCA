
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import cv2

#%% Read in the train.csv file, get indexes

path = "C:/Users/Katon/Documents/train.csv"
train = np.genfromtxt(path, delimiter=",")
# remove top row
train = train[1:,:]
classLabels = train[:,0]

# get indexes
indx0 = np.argwhere(classLabels == 0)[:,0]
indx1 = np.argwhere(classLabels == 1)[:,0]
indx2 = np.argwhere(classLabels == 2)[:,0]
indx3 = np.argwhere(classLabels == 3)[:,0]
indx4 = np.argwhere(classLabels == 4)[:,0]
indx5 = np.argwhere(classLabels == 5)[:,0]
indx6 = np.argwhere(classLabels == 6)[:,0]
indx7 = np.argwhere(classLabels == 7)[:,0]
indx8 = np.argwhere(classLabels == 8)[:,0]
indx9 = np.argwhere(classLabels == 9)[:,0]


# print train shape
print(train.shape)

#%% Create vertical, horizontal, and diagonal masks
vertMask = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

horMask = np.transpose(vertMask)

diagMask = np.zeros((28,28), dtype=int)
diagMask[(vertMask == 0) & (horMask == 0)] = 1
diagMask[22:,22:] = 0
diagMask[25:,19:] = 0
diagMask[19:,25:] = 0

indxD = np.argwhere(diagMask == 1)
indxV = np.argwhere(vertMask == 1)
indxH = np.argwhere(horMask == 1)

# check masks
print("Vertical Mask:\n", vertMask, "\n\n")
print("Horizontal Mask:\n", horMask,"\n\n")
print("Diagonal Mask:\n", diagMask, "\n\n")


#%% DCT

dim = int(np.floor(np.sqrt(train.shape[1])))
numImages = train.shape[0]

dataD = np.empty((indxD.shape[0], numImages))
dataV = np.empty((indxV.shape[0], numImages))
dataH = np.empty((indxH.shape[0], numImages))

for i in range(numImages):
    
    img = np.reshape(train[i, 1:], (dim,dim))
    
    imgDCT = cv2.dct(img)
    imgD = imgDCT * diagMask
    imgV = imgDCT * vertMask
    imgH = imgDCT * horMask
    
    dataD[:,i] = imgD[indxD[:,0], indxD[:,1]]
    dataV[:,i] = imgV[indxV[:,0], indxV[:,1]]
    dataH[:,i] = imgH[indxH[:,0], indxH[:,1]]
    

#%% 
# Take top 2 principal components from each
CD = np.cov(dataD)
pcaD = PCA(n_components=2)
reducedDiag = pcaD.fit_transform(CD)

CV = np.cov(dataV)
pcaV = PCA(n_components=2)
reducedVert = pcaV.fit_transform(CV)

CH = np.cov(dataH)
pcaH = PCA(n_components=2)
reducedHor = pcaH.fit_transform(CH)


pcaFeatures = np.empty((numImages, 6))
pcaFeatures[:,0:2] = np.matmul(dataD.T, pcaD.components_.T)
pcaFeatures[:,2:4] = np.matmul(dataV.T, pcaV.components_.T)
pcaFeatures[:,4:6] = np.matmul(dataH.T, pcaH.components_.T)


#%% plot
ax = plt.gca()
ax.scatter(pcaFeatures[indx0,0], pcaFeatures[indx0,1], s=1, color="red")
ax.scatter(pcaFeatures[indx1,0], pcaFeatures[indx1,1], s=1, color="blue")
ax.scatter(pcaFeatures[indx2,0], pcaFeatures[indx2,1], s=1, color="green")
ax.scatter(pcaFeatures[indx3,0], pcaFeatures[indx3,1], s=1, color="cyan")
ax.scatter(pcaFeatures[indx4,0], pcaFeatures[indx4,1], s=1, color="magenta")
ax.scatter(pcaFeatures[indx5,0], pcaFeatures[indx5,1], s=1, color="yellow")
ax.scatter(pcaFeatures[indx6,0], pcaFeatures[indx6,1], s=1, color="black")
ax.scatter(pcaFeatures[indx7,0], pcaFeatures[indx7,1], s=1, color="violet")
ax.scatter(pcaFeatures[indx8,0], pcaFeatures[indx8,1], s=1, color="orange")
ax.scatter(pcaFeatures[indx9,0], pcaFeatures[indx9,1], s=1, color="pink")


















