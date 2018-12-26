import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import visdom
from tensorboardX import SummaryWriter
import lbtoolbox.util as lbu
import utils as u
import math
import pcl
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

writer = SummaryWriter()
vis = visdom.Visdom()


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, _x):
        return _x.view(self.shape)


class Slot(nn.Module):
    def __init__(self, nin, nout, fs):
        super(Slot, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=nin,
                out_channels=nout,
                kernel_size=(1, fs),
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(nout),
            nn.ReLU()
        )

    def forward(self, _x):
        yy = self.conv1(_x)
        return yy


class Mknet(nn.Module):
    def __init__(self, win_res):
        super(Mknet, self).__init__()
        self.conv2 = nn.Sequential(
            Reshape(-1, 1, 1, win_res),
            Slot(1, 64, 5),
            nn.Dropout2d(0.25),
            Slot(64, 64, 5),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.25),
            Slot(64, 128, 5),
            nn.Dropout2d(0.25),
            Slot(128, 128, 3),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.25),
            Slot(128, 256, 5),
            nn.Dropout2d(0.25)
        )
        self.ConfidenceOutput = nn.Sequential(
            nn.Conv2d(256, 3, (1, 3)),
            Reshape(-1, 3),
            nn.LogSoftmax(dim=-1)
        )

        self.OffsetVoteOutput = nn.Sequential(
            nn.Conv1d(256, 2, (1, 3)),
            Reshape(-1, 2)
        )

    def forward(self, xx):
        yy = self.conv2(xx)
        confi = self.ConfidenceOutput(yy)
        offset = self.OffsetVoteOutput(yy)
        return confi, offset


class bBox2D(object):
    def __init__(self, length, width, xc, yc, theta,
                 alpha,
                 ratio):  # alpha is the bbox's orientation in degrees, theta is the relative angle to the sensor in rad
        self.yc = xc
        self.xc = yc
        self.center=(self.xc,self.yc)
        self.width = length
        self.length = width
        self.theta = theta
        self.alpha = -alpha

    def bBoxCalcVertxex(self, ratio):
        beta = math.atan2(self.width, self.length) * 180 / math.pi
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

        self.vertex1 = self.Rotate(self.vertex1,self.center,self.alpha)
        self.vertex2 = self.Rotate(self.vertex2, self.center, self.alpha)
        self.vertex3 = self.Rotate(self.vertex3, self.center, self.alpha)
        self.vertex4 = self.Rotate(self.vertex4, self.center, self.alpha)

        self.vertex1=(int(self.vertex1[0]*ratio+180),int(self.vertex1[1]*ratio+20))
        self.vertex2=(int(self.vertex2[0]*ratio+180),int(self.vertex2[1]*ratio+20))
        self.vertex3=(int(self.vertex3[0]*ratio+180),int(self.vertex3[1]*ratio+20))
        self.vertex4=(int(self.vertex4[0]*ratio+180),int(self.vertex4[1]*ratio+20))

    def Rotate(self,point,origin,alpha):
        return ((point[0] - origin[0]) * math.cos(alpha*math.pi/180) - (point[1] - origin[1]) * math.sin(alpha*math.pi/180) + origin[0],
        (point[0] - origin[0]) * math.sin(alpha*math.pi/180) + (point[1] - origin[1]) * math.cos(alpha*math.pi/180) + origin[1])


cloudata = np.load('./testset/cloudata.npy')
anndata = np.load('./testset/anndata.npy')
b = torch.FloatTensor(cloudata)
# c=torch.FloatTensor(anndata)
# cv2.namedWindow('scan')
for i, scan in enumerate(cloudata):
    emptyImage = np.zeros([320, 360, 3], np.uint8)
    for dot in scan:
        if dot[0] < 50 and dot[1] < 30 and dot[1] > -30:
            emptyImage[int(dot[0] * 300 / 50 + 20), int(dot[1] * 180 / 30 + 180)] = 255, 255, 0
    for j, label in enumerate(anndata[i]):
        box = bBox2D(label[1], label[2], label[4], label[5], label[7], label[8], 300 / 50)
        #
        # box.bBoxCalcVertxex(300 / 50)
        # cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)

#     outImage = cv2.flip(emptyImage, 0)
#     outImage = cv2.resize(outImage, (1000, 1000), interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('scan', outImage)
#     cv2.waitKey()
# cv2.destroyAllWindows()
# print(b.size(), '\t')

trainset=data.TensorDataset


