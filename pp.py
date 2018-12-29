# import os
# import pcl
import numpy as np
import torch
import cv2
import math


# import  pretrainedmodels


# class Point(object):
#     def __init__(self,x,y,z):
#         self.x = x
#         self.y = y
#         self.z = z
# points = []

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


# pathbox = 'D:/JupyterNotebook/testset/_bbox/'
# pathpcd = 'D:/JupyterNotebook/testset/pcd/'
#
# for fpathe, dirs, fs in os.walk(pathbox):
#     cloudata=[]
#     for f in fs:
#         # print(os.path.join(fpathe, f))
#         pcdname=f.replace('.txt','.pcd')
#         print(pcdname)
#         p = pcl.PointCloud()
#         p.from_file(pathpcd+pcdname)
#         parray=p.to_array()
#         cloudata.append(parray)
#
#     # cloudtensor=torch.FloatTensor(cloudata)
#     print(len(cloudata))
#     np.save('D:/JupyterNotebook/testset/cloudata',cloudata)

cloudata = np.load('./testset/cloudata.npy')
anndata = np.load('./testset/anndata.npy')
b = torch.FloatTensor(cloudata)
img = []
# c=torch.FloatTensor(anndata)
# cv2.namedWindow('scan')
for i, scan in enumerate(cloudata):
    emptyImage = np.zeros([200, 180, 3], np.uint8)
    for dot in scan:
        if dot[0] < 30 and dot[1] < 15 and dot[1] > -15:
            emptyImage[int(dot[0] * 180 / 30 + 20), int(dot[1] * 90 / 15 + 90)] = (
            int(math.hypot(dot[0], dot[1]) * 255 / 60), int(dot[0] * 235 / 30 + 20), int(dot[1] * 75 / 15 + 180))
    for j, label in enumerate(anndata[i]):
        if label[0] < label[1] and (label[4] == -90 or label[4] == 0 or label[4] == 90 or label[4] == -180):
            box = bBox2D(label[0], label[1], label[3], label[2], -label[4], 300 / 50)  # fix annotations!!!
        # elif label[4]==90 or label[4]==-90:
        #     box = bBox2D(label[0], label[1], label[2], label[3], label[4], 300 / 50)
        else:
            box = bBox2D(label[1], label[0], label[3], label[2], -label[4], 300 / 50)

        box.Scale(300 / 50, 90, 20)
        box.bBoxCalcVertxex()
        anndata[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]

        # cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)

    # outImage = cv2.flip(emptyImage, 0)
    # outImage = cv2.flip(outImage, 1)
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

#
# filename = 'D:/1544600733.580758018'
#
# p = pcl.PointCloud()
# p.from_file(filename+'.pcd')
# print(p)
# p.to_file("ppp.txt")
# a=p.to_array()
# print(a)
# with open(filename+'.pcd','r',encoding='UTF-8') as f:
#     for line in  f.readlines()[11:len(f.readlines())-1]:
#         strs = line.split(' ')
#         points.append(Point(strs[0],strs[1],strs[2].strip()))

# fw = open(filename+'.txt','w')
# for i in range(len(points)):
#      linev = points[i].x+" "+points[i].y+" "+points[i].z+"\n"
#      fw.writelines(linev)
# fw.close()
