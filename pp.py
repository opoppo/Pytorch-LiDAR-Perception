# import os
# import pcl
import numpy as np
import torch
import cv2
import math
from bBox_2D import bBox_2D

cloudata = np.load('./testset/cloudata.npy')
anndata = np.load('./testset/anndata.npy')
# b = torch.FloatTensor(cloudata)
img = []

# ==============================
resolution = 224  # res*res !!!   (224 ResNet  299 Inception  1000 Visualization ONLY)
# ==============================

for i, scan in enumerate(cloudata):
    emptyImage = np.zeros([200, 200, 3], np.uint8)
    for dot in scan:
        if dot[0] < 30 and 100 / 6 > dot[1] > -100 / 6:
            emptyImage[int(dot[0] * 180 / 30 + 20), int(dot[1] * 6 + 100)] = (
                int(255 - math.hypot(dot[0], dot[1]) * 255 / 60), int(255 - (dot[0] * 235 / 30 + 20)),
                int(dot[1] * 75 / 15 + 80))
    for j, label in enumerate(anndata[i]):
        if label[0] < label[1] and (label[4] == -90 or label[4] == 0 or label[4] == 90 or label[4] == -180):
            box = bBox_2D(label[0], label[1], label[3], label[2], -label[4])  # fix annotations!!!
        else:
            box = bBox_2D(label[1], label[0], label[3], label[2], -label[4])

        box.scale(300 / 50, 100, 20)
        box.scale(resolution / 200, 0, 0)  # ===== !!!
        # box.Scale(299 / 200, 0, 0)  # ===== !!!
        box.bBoxCalcVertxex()
        anndata[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]

        # cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)

    outImage = cv2.resize(emptyImage, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('scan', outImage)
    img.append(outImage)
    print(i)
    # cv2.waitKey()
# cv2.destroyAllWindows()
# TODO: data augmentation, noise


# Flipping
augmentimg = []
for i, im in enumerate(img):
    imflipped = cv2.flip(im, 1)
    augmentimg.append(imflipped)
img = img + augmentimg
del augmentimg

augmentann = np.zeros(anndata.shape, dtype=np.float)
for i, scan in enumerate(anndata):
    for j, label in enumerate(scan):
        box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
        box.flipx(axis=int(resolution / 2))
        augmentann[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]
anndata = np.concatenate((anndata, augmentann))
del augmentann

# noise to rotate, translate(x,y), resize
print('Adding Noise...')
augmentann = np.zeros(anndata.shape, dtype=np.float)
for i, scan in enumerate(anndata):
    for j, label in enumerate(scan):
        noiseratio = ((torch.randn(2)).div_(20)).exp_()
        noiseoffset = (torch.randn(2))
        box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
        box.rotate(noiseratio[0])
        box.resize(noiseratio[1])
        box.translate(noiseoffset[0], noiseoffset[1])
        augmentann[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]
anndata = np.concatenate((anndata, augmentann))
del augmentann
img = img + img

print(cloudata.shape, '\t', anndata.shape, '\t', len(img))
np.save('./testset/img', img)
np.save('./testset/anndatafixed', anndata)
