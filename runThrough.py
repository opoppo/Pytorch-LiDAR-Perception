import os
# import pcl
import numpy as np
import torch
import cv2
import math
import  torchvision
import torch.utils.data as data
import torch.nn as nn
import time


# class Point(object):
#     def __init__(self,x,y,z):
#         self.x = x
#         self.y = y
#         self.z = z
# points = []

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

        self.vertex1=(int(self.vertex1[0]*ratio+90),int(self.vertex1[1]*ratio+20))
        self.vertex2=(int(self.vertex2[0]*ratio+90),int(self.vertex2[1]*ratio+20))
        self.vertex3=(int(self.vertex3[0]*ratio+90),int(self.vertex3[1]*ratio+20))
        self.vertex4=(int(self.vertex4[0]*ratio+90),int(self.vertex4[1]*ratio+20))

    def Rotate(self,point,origin,alpha):
        return ((point[0] - origin[0]) * math.cos(alpha*math.pi/180) - (point[1] - origin[1]) * math.sin(alpha*math.pi/180) + origin[0],
        (point[0] - origin[0]) * math.sin(alpha*math.pi/180) + (point[1] - origin[1]) * math.cos(alpha*math.pi/180) + origin[1])

# class train_data_set(data.Dataset):
#     def __init__(self, DataTensor, Target):
#         self.DataTensor = DataTensor
#         self.Target= Target
#
#     def __getitem__(self, index):
#         return self.DataTensor[index], self.Target[index]
#
#     def __len__(self):
#         return self.DataTensor.size(0)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, _x):
        return _x.view(self.shape)

class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        # self.fc = nn.Linear(2048, (5,5), bias=True)
        self.fc=nn.Sequential(
            nn.Linear(2048, 25, bias=True),
            Reshape(-1,5,5)
        )
    def forward(self, X):
        y = self.fc(X)
        # print(y.size())
        return y

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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

# cloudata = np.load('./testset/cloudata.npy')
# anndata = np.load('./testset/anndata.npy')
# b = torch.FloatTensor(cloudata)
# img=[]
# c=torch.FloatTensor(anndata)
# cv2.namedWindow('scan')
# for i, scan in enumerate(cloudata):
#     emptyImage = np.zeros([200, 180, 3], np.uint8)
#     for dot in scan:
#         if dot[0] < 30 and dot[1] < 15 and dot[1] > -15:
#             emptyImage[int(dot[0] * 180 / 30 + 20), int(dot[1] * 90 / 15 + 90)] = (int(math.hypot(dot[0],dot[1])*255/60), int(dot[0]*235/30+20), int(dot[1]*75/15+180))
#     for j, label in enumerate(anndata[i]):
#         if label[1]>=label[2]:
#             box = bBox2D(label[1], label[2], label[4], label[5], label[7], label[8], 300 / 50)
#         else:
#             box = bBox2D(label[2], label[1], label[4], label[5], label[7], label[8], 300 / 50)
#
#         box.bBoxCalcVertxex(300 / 50)
#         cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
#         cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
#         cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
#         cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)
#
#     outImage = cv2.flip(emptyImage, 0)
#     outImage = cv2.flip(outImage, 1)
#     outImage = cv2.resize(outImage, (1000, 1000), interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('scan', outImage)
#     img.append(outImage)
#     print(i)
#     cv2.waitKey()
# cv2.destroyAllWindows()
# print(b.size(), '\t')

# np.save('./testset/img',img)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

img=np.load('./testset/img.npy')
imgtensor=torch.FloatTensor(img)
imgtensor=imgtensor.permute(0,3,2,1)
print(imgtensor.size())
del img
anndata = np.load('./testset/anndata.npy')
anntensor=torch.FloatTensor(anndata).cuda()
del anndata

net=torchvision.models.resnet101(pretrained=True)

net.fc = OutputLayer()
net=torch.nn.DataParallel(net.cuda(),device_ids=[0,1,2,3])
trainset=data.TensorDataset(imgtensor,anntensor)
train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=256,  #256 for 4 GPUs
    shuffle=True,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
)

optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
mseloss=nn.MSELoss(reduction='sum')
lambda1=lambda epoch: 10**np.random.uniform(-3,-6)
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda1)


# ====================================================================================================
training = 1  # ????========================================================================================
# ====================================================================================================

# net=torch.load('nettmp')
# # print('nettmp loaded')

# Predicting
if (not training):
    net=torch.load('nettt')
    print('nettt loaded')

# training
if training:

    EPOCH = 1000
    break_flag = False
    prevvalloss, prevtrainloss = 9999, 9999
    ppp = 0
    waitfor = 5  # rounds to wait for further improvement before quit training=================================

    totaltime, losslist = [], []

    for epoch in range(EPOCH):

        scheduler.step()

        if break_flag is True:
            break

        net.train()
        time_start = time.time()
        epochTloss= 0

        for step, (x, bboxes) in enumerate(train_loader):
            # print(bboxes.size()," ===")
            if break_flag is True:
                break

            bboxes_out = net(x)
            # print(bboxes_out.size())
            del x
            # torch.Size([15360, 3]) torch.Size([15360, 2])
            # yConfl = yConf.type(torch.cuda.LongTensor)
            #
            # tgt_noise = ((torch.randn(*yOffs.shape)).div_(20)).exp_().type(torch.cuda.FloatTensor)
            # mask = ((yConf != 0).view((-1, 1))).type(torch.cuda.FloatTensor)  # Tensor Type match!!!!
            # del yConf
            # n = mask.sum()

            # a = loss(outConf.mul_((1 - outConf.exp()).pow_(2)), yConfl)  # Focal loss
            # if n > 0:
            #     b = ((loss(outOffs.mul_(mask), yOffs.mul_(tgt_noise))).div_(
            #         n)).sqrt_()  # RMSE loss
            # else:
            #     b = (loss(outOffs.mul_(mask), yOffs)).sqrt_()
            # del tgt_noise, mask, n, yConfl, yOffs

            # print(outConf.shape, yConfl.shape,mask.shape) #torch.Size([15360, 3]) torch.Size([15360]) which is correct
            loss = mseloss(bboxes_out,bboxes)
            epochTloss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_end = time.time()
        totaltime.append(time_end - time_start)
        losslist.append((epoch, epochTloss))
        lr=get_lr(optimizer)
        print("EPOCH", epoch, "  loss_total: %.4f" % epochTloss, "  epoch_time: %.2f" % (time_end - time_start),
              "s   estimated_time: %.2f" % ((EPOCH - epoch - 1) * sum(totaltime) / ((epoch + 1) * 60)), "min with lr=",lr)

        torch.save(net, "nettmp")
        # print("===new model saved===")


torch.save(net, "nettt")
print("====final model saved====")
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
