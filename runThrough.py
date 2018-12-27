import os
import numpy as np
import torch
import cv2
import math
import torchvision
import torch.utils.data as data
import torch.nn as nn
import time


class bBox2D(object):
    def __init__(self, length, width, xc, yc, theta,
                 alpha,
                 ratio):  # alpha is the bbox's orientation in degrees, theta is the relative angle to the sensor in rad
        self.yc = xc
        self.xc = yc
        self.center = (self.xc, self.yc)
        self.width = length
        self.length = width
        self.theta = theta
        self.alpha = -alpha

    def bBoxCalcVertxex(self, ratio):
        self.vertex1 = (self.xc + self.length / 2, self.yc + self.width / 2)
        self.vertex2 = (self.xc + self.length / 2, self.yc - self.width / 2)
        self.vertex3 = (self.xc - self.length / 2, self.yc + self.width / 2)
        self.vertex4 = (self.xc - self.length / 2, self.yc - self.width / 2)

        self.vertex1 = self.Rotate(self.vertex1, self.center, self.alpha)
        self.vertex2 = self.Rotate(self.vertex2, self.center, self.alpha)
        self.vertex3 = self.Rotate(self.vertex3, self.center, self.alpha)
        self.vertex4 = self.Rotate(self.vertex4, self.center, self.alpha)

        self.vertex1 = (int(self.vertex1[0] * ratio + 90), int(self.vertex1[1] * ratio + 20))
        self.vertex2 = (int(self.vertex2[0] * ratio + 90), int(self.vertex2[1] * ratio + 20))
        self.vertex3 = (int(self.vertex3[0] * ratio + 90), int(self.vertex3[1] * ratio + 20))
        self.vertex4 = (int(self.vertex4[0] * ratio + 90), int(self.vertex4[1] * ratio + 20))

    def Rotate(self, point, origin, alpha):
        return ((point[0] - origin[0]) * math.cos(alpha * math.pi / 180) - (point[1] - origin[1]) * math.sin(
            alpha * math.pi / 180) + origin[0],
                (point[0] - origin[0]) * math.sin(alpha * math.pi / 180) + (point[1] - origin[1]) * math.cos(
                    alpha * math.pi / 180) + origin[1])


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
        self.fc = nn.Sequential(
            nn.Linear(2048, 60, bias=True),
            Reshape(-1, 12, 5)
        )

    def forward(self, X):
        y = self.fc(X)
        # print(y.size())
        return y


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration / (2 * stepsize))
    x = np.abs(iteration / stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

img = np.load('./testset/img.npy')
imgtensor = torch.FloatTensor(img)
imgtensor = imgtensor.permute(0, 3, 2, 1)
print(imgtensor.size())
del img
anndata = np.load('./testset/anndata.npy')
anntensor = torch.FloatTensor(anndata).cuda()
del anndata

net = torchvision.models.resnet101(pretrained=True)

net.fc = OutputLayer()
net = torch.nn.DataParallel(net.cuda(), device_ids=[0, 1, 2, 3])
trainset = data.TensorDataset(imgtensor, anntensor)
train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=256,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
    sampler=data.SubsetRandomSampler(list(range(0, 3000, 1)))
)
val_loader = data.DataLoader(
    dataset=trainset,
    batch_size=256,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
    sampler=data.SubsetRandomSampler(list(range(3000, 3500, 1)))
)
test_loader = data.DataLoader(
    dataset=trainset,
    batch_size=256,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
    sampler=data.SubsetRandomSampler(list(range(3500, 4239, 1)))
)

optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
mseloss = nn.MSELoss(reduction='sum')
# lambda1=lambda epoch: 10**np.random.uniform(-3,-6)
lambda1 = lambda epoch: get_triangular_lr(epoch, 30, 10 ** (-2), 10 ** (0))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

# ====================================================================================================
training = 1  # ????========================================================================================
# ====================================================================================================

# net = torch.load('nettt')
# print('nettmp loaded')#===================

# Predicting
if (not training):
    net = torch.load('nettt')
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
        epochTloss = 0

        for step, (x, bboxes) in enumerate(train_loader):
            # print(bboxes.size()," ===")
            if break_flag is True:
                break

            bboxes_out = net(x)
            # print(bboxes_out.size())
            del x

            # print(bboxes_out.size(),bboxes.size())
            loss = mseloss(bboxes_out, bboxes)
            epochTloss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_end = time.time()
        totaltime.append(time_end - time_start)
        losslist.append((epoch, epochTloss))
        lr = get_lr(optimizer)
        print("EPOCH", epoch, "  loss_total: %.4f" % epochTloss, "  epoch_time: %.2f" % (time_end - time_start),
              "s   estimated_time: %.2f" % ((EPOCH - epoch - 1) * sum(totaltime) / ((epoch + 1) * 60)), "min with lr=%e"
              % lr)

        # torch.save(net, "nettmp")
        # print("===new model saved===")
        if (epoch + 1) % 5 == 0:

            net.eval()
            epochvalloss = 0

            for step, (x, bboxes) in enumerate(val_loader):
                bboxes_out = net(x)
                del x

                loss = mseloss(bboxes_out, bboxes)
                epochvalloss += loss.item()

            print("loss_total: %.4f" % epochvalloss, " on validation")

            # writer.add_scalar('data/valloss', epochvalloss, epoch)
            if epochvalloss <= prevvalloss and epochTloss <= prevtrainloss:
                torch.save(net, "nettmp")
                print("===improved model saved===")
                prevtrainloss = epochTloss
                prevvalloss = epochvalloss
                ppp = 0
            else:
                ppp += 1
                print("===tried round ", ppp, " ===")
                if ppp >= waitfor:
                    net = torch.load('nettmp')
                    print("===dead end, rolling back to previous model===")
                    break_flag = True

    torch.save(losslist, "losslist.pt")

#

net.eval()
result = []

epochtestloss = 0

for step, (x, bboxes) in enumerate(test_loader):
    bboxes_out = net(x)
    del x

    loss = mseloss(bboxes_out, bboxes)
    epochtestloss += loss.item()

print("loss_total: %.4f" % epochtestloss, " on testset")

torch.save(result, "result.pt")
torch.save(net, "nettt")
print("====final model saved====")
