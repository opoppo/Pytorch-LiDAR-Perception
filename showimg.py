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
        # beta = math.atan2(self.width, self.length) * 180 / math.pi
        # gamma = beta - self.alpha
        # gamma1 = beta + self.alpha
        # r = math.sqrt((self.width / 2) ** 2 + (self.length / 2) ** 2)
        # self.vertex1 = (int(self.xc + (r * math.cos(gamma * math.pi / 180)*ratio)),int(self.yc + (r * math.sin(gamma * math.pi / 180)*ratio)))
        # self.vertex2 = (int(self.xc + (r * math.cos(gamma1 * math.pi / 180)*ratio)), int(self.yc + (r * math.sin(gamma1* math.pi / 180)*ratio)))
        # self.vertex3 = (int(self.xc - (r * math.cos(gamma * math.pi / 180)*ratio)), int(self.yc - (r * math.sin(gamma * math.pi / 180)*ratio)))
        # self.vertex4 = (int(self.xc - (r * math.cos(gamma1 * math.pi / 180)*ratio)), int(self.yc - (r * math.sin(gamma1 * math.pi / 180)*ratio)))
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


img = np.load('./testset/img.npy')
anndata = np.load('./testset/anndatafixed.npy')

for i, im in enumerate(img):
    emptyImage = cv2.resize(im, (180, 200), interpolation=cv2.INTER_CUBIC)
    del im
    for j, label in enumerate(anndata[i]):
        box = bBox2D(label[0], label[1], label[2], label[3], label[4], 300 / 50)
        # box.Scale(1,90/6,20/6)
        box.bBoxCalcVertxex()
        cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)
    outImage = cv2.resize(emptyImage, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('scan', outImage)
    cv2.waitKey()
