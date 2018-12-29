# import os
# import pcl
import numpy as np
import torch
import cv2
import math


class bBox2D(object):
    def __init__(self, length, width, xc, yc,
                 alpha,
                 ratio):  # alpha is the bbox's orientation in degrees, theta is the relative angle to the sensor in rad
        self.yc = yc
        self.xc = xc
        self.center = (self.xc, self.yc)
        self.width = width
        self.length = length
        # self.theta = theta
        self.alpha = alpha

    def bBoxCalcVertxex(self):
        self.vertex1 = (self.xc + self.length / 2, self.yc + self.width / 2)
        self.vertex2 = (self.xc + self.length / 2, self.yc - self.width / 2)
        self.vertex3 = (self.xc - self.length / 2, self.yc + self.width / 2)
        self.vertex4 = (self.xc - self.length / 2, self.yc - self.width / 2)

        self.vertex1 = self.Rotate(self.vertex1, self.center, self.alpha)
        self.vertex2 = self.Rotate(self.vertex2, self.center, self.alpha)
        self.vertex3 = self.Rotate(self.vertex3, self.center, self.alpha)
        self.vertex4 = self.Rotate(self.vertex4, self.center, self.alpha)

        self.vertex1 = (int(self.vertex1[0]), int(self.vertex1[1]))
        self.vertex2 = (int(self.vertex2[0]), int(self.vertex2[1]))
        self.vertex3 = (int(self.vertex3[0]), int(self.vertex3[1]))
        self.vertex4 = (int(self.vertex4[0]), int(self.vertex4[1]))

    def Scale(self, ratio, offsx, offsy):
        self.yc = self.yc * ratio + offsy
        self.xc = self.xc * ratio + offsx
        self.center = (self.xc, self.yc)
        self.width = self.width * ratio
        self.length = self.length * ratio

    def Rotate(self, point, origin, alpha):
        return ((point[0] - origin[0]) * math.cos(alpha * math.pi / 180) - (point[1] - origin[1]) * math.sin(
            alpha * math.pi / 180) + origin[0],
                (point[0] - origin[0]) * math.sin(alpha * math.pi / 180) + (point[1] - origin[1]) * math.cos(
                    alpha * math.pi / 180) + origin[1])


cloudata = np.load('./testset/cloudata.npy')
anndata = np.load('./testset/anndata.npy')
b = torch.FloatTensor(cloudata)
img = []

for i, scan in enumerate(cloudata):
    emptyImage = np.zeros([200, 200, 3], np.uint8)
    for dot in scan:
        if dot[0] < 30 and dot[1] < 100 / 6 and dot[1] > -100 / 6:
            emptyImage[int(dot[0] * 180 / 30 + 20), int(dot[1] * 6 + 100)] = (
                int(255 - math.hypot(dot[0], dot[1]) * 255 / 60), int(255 - (dot[0] * 235 / 30 + 20)),
                int(dot[1] * 75 / 15 + 80))
    for j, label in enumerate(anndata[i]):
        if label[0] < label[1] and (label[4] == -90 or label[4] == 0 or label[4] == 90 or label[4] == -180):
            box = bBox2D(label[0], label[1], label[3], label[2], -label[4], 300 / 50)  # fix annotations!!!
        else:
            box = bBox2D(label[1], label[0], label[3], label[2], -label[4], 300 / 50)

        box.Scale(300 / 50, 100, 20)
        box.Scale(299 / 200, 0, 0)  # ===== !!!
        box.bBoxCalcVertxex()
        anndata[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]

        # cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)

    # outImage = cv2.resize(emptyImage, (224, 224), interpolation=cv2.INTER_CUBIC)  # ResNet
    outImage = cv2.resize(emptyImage, (299, 299), interpolation=cv2.INTER_CUBIC)  # Inception
    # outImage = cv2.resize(emptyImage, (1000, 1000), interpolation=cv2.INTER_CUBIC)   # Visualization ONLY!
    # cv2.imshow('scan', outImage)
    img.append(outImage)
    print(i)
    # cv2.waitKey()
# cv2.destroyAllWindows()
print(b.size(), '\t')
# TODO: data augmentation, noise
np.save('./testset/img', img)
np.save('./testset/anndatafixed', anndata)
