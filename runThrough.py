import os
import numpy as np
import torch
import cv2
import torchvision
import torch.utils.data as data
import torch.nn as nn
import time
import pretrainedmodels
from bBox_2D import bBox_2D


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
            Reshape(-1, 3, 5)
        )

    def forward(self, X):
        y = self.fc(X)
        # print(y.size())
        return y


class OutputLayerInceptionv4(nn.Module):
    def __init__(self):
        super(OutputLayerInceptionv4, self).__init__()
        # self.fc = nn.Linear(2048, (5,5), bias=True)
        self.fc = nn.Sequential(
            nn.Linear(1536, 60, bias=True),
            Reshape(-1, 3, 5)
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


# ====================================================================================================
training = 1  # ????========================================================================================
resume = 0  # ====010:  test model   11X: train model   10X: train new   011: refresh dataset
generateNewSets = 0  # REGENERATE the datasets !!!!!!!!!!!!!!!
# ====================================================================================================


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

img = np.load('./testset/img.npy')
imgtensor = torch.FloatTensor(img)
imgtensor = imgtensor.permute(0, 3, 2, 1)
print(imgtensor.size())
del img
anndata = np.load('./testset/anndatafixed.npy')
anntensor = torch.FloatTensor(anndata).cuda()
del anndata

if training and not resume:
    # net = torchvision.models.resnet101(pretrained=True)   #256  10s
    # net.fc = OutputLayer()
    net = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')  # 156  20s
    net.last_linear = OutputLayer()
    # net = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')  # 186  20s
    # net.last_linear = OutputLayerInceptionv4()
    net = torch.nn.DataParallel(net.cuda(), device_ids=[0, 1, 2, 3])

rawset = data.TensorDataset(imgtensor, anntensor)

if generateNewSets:
    (trainset, valset, testset) = data.random_split(rawset, [int(len(rawset) * 0.70), int(len(rawset) * 0.15),
                                                             len(rawset) - int(len(rawset) * 0.70) - int(
                                                                 len(rawset) * 0.15)])
    torch.save(trainset, './testset/dataset/trainset')
    torch.save(valset, './testset/dataset/valset')
    torch.save(testset, './testset/dataset/testset')
else:
    trainset = torch.load('./testset/dataset/trainset')
    valset = torch.load('./testset/dataset/valset')
    testset = torch.load('./testset/dataset/testset')

print(len(trainset), len(valset), len(testset))

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=136,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12,
    # sampler=data.SubsetRandomSampler(list(range(0, 3000, 1)))
)
val_loader = data.DataLoader(
    dataset=valset,
    batch_size=4,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
    # sampler=data.SubsetRandomSampler(list(range(3000, 3500, 1)))
)
test_loader = data.DataLoader(
    dataset=testset,
    batch_size=1,  # 256 for 4 GPUs
    shuffle=False,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
    # sampler=data.SubsetRandomSampler(list(range(3500, 4239, 1)))
)

if resume and training:
    net = torch.load('net-0.2')
    print('net resumed')  # ==============================================================

# Predicting or Testing============
if resume and (not training):
    net = torch.load('net-0.2')
    print('nettt loaded')

optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, weight_decay=0.001)
mseloss = nn.MSELoss(reduction='mean')
# lambda1=lambda epoch: 10**np.random.uniform(-3,-6)
lambda1 = lambda epoch: get_triangular_lr(epoch, 100, 10 ** (-3), 10 ** (0))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

# training
if training:

    EPOCH = 1000
    break_flag = False
    prevvalloss, prevtrainloss = 10e30, 10e30
    ppp = 0
    waitfor = 5  # rounds to wait for further improvement before quit training=================================

    totaltime, losslist = [], []

    for epoch in range(EPOCH):

        net.train()
        scheduler.step()
        time_start = time.time()
        epochTloss = 0
        if break_flag is True:
            break

        for step, (x, bboxes) in enumerate(train_loader):
            if break_flag is True:
                break

            bboxes_out = net(x)
            del x

            print(bboxes_out.size(),bboxes.size())
            loss = mseloss(bboxes_out, bboxes)
            # epochTloss += loss.item()
            epochTloss = loss.item()

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

        if (epoch + 1) % 25 == 0:

            net.eval()
            epochvalloss = 0

            for step, (x, bboxes) in enumerate(val_loader):
                bboxes_out = net(x)
                del x

                loss = mseloss(bboxes_out, bboxes)
                # epochvalloss += loss.item()
                epochvalloss = loss.item()

            print("loss_total: %.4f" % epochvalloss, " on validation")

            if epochvalloss <= prevvalloss and epochTloss <= prevtrainloss:
                torch.save(net, "nettmp")
                print("===improved model saved===")
                prevtrainloss = epochTloss
                prevvalloss = epochvalloss
                ppp = 0
            # else:
            #     ppp += 1
            #     print("===tried round ", ppp, " ===")
            #     if ppp >= waitfor:
            #         net = torch.load('nettmp')
            #         print("===dead end, rolling back to previous model===")
            #         break_flag = True

    torch.save(losslist, "losslist.pt")

#
net.eval()
result = []
epochtestloss = 0

for step, (x, bboxes) in enumerate(test_loader):
    bboxes_out = net(x)
    # print(bboxes_out.size(),bboxes.size())
    x = x.squeeze_().permute(2, 1, 0)
    emptyImage = x.cpu().detach().numpy().copy()
    # print(emptyImage.shape,type(emptyImage))
    # emptyImage = cv2.resize(emptyImage, (200, 200), interpolation=cv2.INTER_CUBIC)

    del x
    for j, label in enumerate(bboxes.squeeze_().detach().cpu().numpy()):
        box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
        # box.Scale(300 / 50, 100, 20)
        box.bBoxCalcVertxex()
        cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)

    for j, label in enumerate(bboxes_out.squeeze_().detach().cpu().numpy()):
        box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
        # box.Scale(300 / 50, 100, 20)
        # box.Scale(299 / 200, 0, 0)
        box.bBoxCalcVertxex()
        cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 55), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 55), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 55), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 55), 1, cv2.LINE_AA)

    emptyImage = cv2.flip(emptyImage, 0)
    emptyImage = cv2.flip(emptyImage, 1)
    outImage = cv2.resize(emptyImage, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('scan', outImage)
    print(step)
    cv2.imwrite('./testset/Result/%d.jpg' % step, outImage)
    # cv2.waitKey()

    loss = mseloss(bboxes_out, bboxes)
    epochtestloss = loss.item()

print("loss_total: %.4f" % epochtestloss, " on testset")

torch.save(result, "result.pt")
torch.save(net, "nettt")
print("====final model saved====")
