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
from ignite.engine import  Events, create_supervised_trainer, create_supervised_evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

writer = SummaryWriter()
vis = visdom.Visdom()


def calcPreRec(_X, _scans, _wcs, _was, _r, ts):
    mylist1 = []
    mylist2 = []
    for Xb in lbu.batched(1, _X, shuf=False, droplast=True):
        Xbt = torch.Tensor(Xb).cuda(non_blocking=True)
        del Xb
        pred_y_conf0, pred_y_offs0 = net(Xbt)
        del Xbt
        pred_y_conf, pred_y_offs = pred_y_conf0.detach().cpu().numpy(), pred_y_offs0.detach().cpu().numpy()
        del pred_y_offs0, pred_y_conf0
        mylist1.append(pred_y_offs)
        mylist2.append(pred_y_conf)
        del pred_y_conf, pred_y_offs

    pred_y_conf_batches = np.array(mylist2)
    pred_y_offs_batches = np.array(mylist1)
    del mylist1, mylist2
    # print(pred_y_conf_batches.shape)
    alldets, (wcdets, wadets) = pred2det_comb(_scans, pred_y_conf_batches, pred_y_offs_batches, thresh=ts,
                                              out_rphi=False)
    gt_all = [wcs + was for wcs, was in zip(_wcs, _was)]
    precs, recs = u.precrec(alldets, gt_all, _r, pred_rphi=False, gt_rphi=True)
    return precs, recs


def pred2det_comb(scans, confs, offss, thresh, out_rphi=True):
    K = confs.shape[2] - 1  # .shape[2] - 1  # Minus the "nothing" type.
    dets_all = []  # All detections combined, regardless of type.
    dets = [[] for _ in range(K)]  # All detections of each type

    for scan, conf, offs in zip(scans, confs, offss):
        dall, d = [], [[] for _ in range(K)]

        locs, probs = pred2votes(scan, conf, offs, thresh=thresh)
        for x, y, lbl in u.votes_to_detections(locs, probs, in_rphi=False, out_rphi=out_rphi):
            dall.append((x, y))
            d[lbl].append((x, y))
        dets_all.append(dall)
        for k in range(K):
            dets[k].append(d[k])

    return dets_all, dets


def pred2votes(scan, y_conf, y_offs, thresh):
    locs, probs = [], []
    for (pno, pwc, pwa), (dx, dy), r, phi in zip(y_conf, y_offs, scan, u.laser_angles(len(scan))):
        # print(math.exp(pno),"   ",math.exp(pwc),"   ",math.exp(pwa))
        if thresh < np.exp(pwc) + np.exp(pwa):
            locs.append(u.rphi_to_xy(*win2global(r, phi, dx, dy)))
            probs.append((np.exp(pwc), np.exp(pwa)))
    return locs, probs


def win2global(r, phi, dx, dy):
    y = r + dy
    dphi = np.arctan2(dx, y)  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    return y / np.cos(dphi), phi + dphi


def CalcAP(precslist, recslist):
    list.sort(precslist, reverse=True)
    list.sort(recslist, reverse=False)
    acum_area = 0
    prevpr, prevre = 1, 0
    for pr, re in zip(precslist, recslist):
        if re[0] > prevre and (not math.isnan(pr[0])) and (not math.isnan(re[0])):
            acum_area += 0.5 * (pr[0] + prevpr) * (re[0] - prevre)
            prevpr = pr[0]
            prevre = re[0]
    return acum_area


def compute_precrecs(
        scans, pred_conf, pred_offs, gt_wcs, gt_was,
        ts=(1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1 - 1e-3, 1 - 1e-5),
        rs=(0.1, 0.3, 0.5, 0.7, 0.9)
):
    gt_all = [wcs + was for wcs, was in zip(gt_wcs, gt_was)]

    if isinstance(rs, (float, int)):
        rs = (rs,)

    mkgt = lambda: np.full((len(ts), len(rs)), np.nan)
    precs, recs = mkgt(), mkgt()
    precs_wc, recs_wc = mkgt(), mkgt()
    precs_wa, recs_wa = mkgt(), mkgt()

    for i, t in enumerate(ts):
        alldets, (wcdets, wadets) = pred2det_comb(scans, pred_conf, pred_offs, thresh=t, out_rphi=False)
        for j, r in enumerate(rs):
            precs[i, j], recs[i, j] = u.precrec(alldets, gt_all, r, pred_rphi=False, gt_rphi=True)
            precs_wc[i, j], recs_wc[i, j] = u.precrec(wcdets, gt_wcs, r, pred_rphi=False, gt_rphi=True)
            precs_wa[i, j], recs_wa[i, j] = u.precrec(wadets, gt_was, r, pred_rphi=False, gt_rphi=True)
        lbu.printnow(".")
    return (precs, recs), (precs_wc, recs_wc), (precs_wa, recs_wa)


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


Xtr = np.load("./Xtr.npy")
# Xtr = torch.load('Xtrt.pt')
ytr_conf = torch.load('ytr_conf.pt')
ytr_offs = torch.load('ytr_offs.pt')
yva_conf = torch.load('yva_conf.pt')
yva_offs = torch.load('yva_offs.pt')
Xva = torch.load('Xva.pt')
Xte = torch.load('Xte.pt')

va_wcs = torch.load('va_wcs.pt')
te_wcs = torch.load('te_wcs.pt')
tr_wcs = torch.load('tr_wcs.pt')

va_was = torch.load('va_was.pt')
te_was = torch.load('te_was.pt')
tr_was = torch.load('tr_was.pt')

va_scans = torch.load('va_scans.pt')
te_scans = torch.load('te_scans.pt')
tr_scans = torch.load('tr_scans.pt')

# Setting up
DataTensor = torch.cuda.FloatTensor(Xtr)
OffsTargetTensor = torch.cuda.FloatTensor(ytr_offs)
ConfTargetTensor = torch.cuda.FloatTensor(ytr_conf)
del ytr_conf, ytr_offs, Xtr

trainset = data.TensorDataset(DataTensor, OffsTargetTensor, ConfTargetTensor)

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=60000,  # 120000 for 4 GPUs ; 80000 for 3 GPUs
    shuffle=True,
    drop_last=True,
    # pin_memory=True,
    # num_workers=12
)
TestTensor = torch.cuda.FloatTensor(Xte)

testset = data.TensorDataset(TestTensor)

test_loader = data.DataLoader(
    dataset=testset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    # pin_memory=True
)

DataTensor1 = torch.cuda.FloatTensor(Xva).view((-1,48))
OffsTargetTensor1 = torch.cuda.FloatTensor(yva_offs)
ConfTargetTensor1 = torch.cuda.FloatTensor(yva_conf)
del yva_conf, yva_offs, Xva

valset = data.TensorDataset(DataTensor1, OffsTargetTensor1, ConfTargetTensor1)

val_loader = data.DataLoader(
    dataset=valset,
    batch_size=100,  # 120000 for 4 GPUs ; 80000 for 3 GPUs
    shuffle=True,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
)


# net = Mknet(win_res=48)#=====================================

# net=torchvision.models.resnet50(pretrained=True)
# net.conv1=(3,)

# net=torch.nn.DataParallel(net.cuda(),device_ids=[0,1])
net = torch.load('net-10')
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
loss_class = nn.NLLLoss(weight=torch.Tensor((0.5, 10, 10)).cuda(non_blocking=True))
loss_offset = nn.MSELoss(reduction='sum')
loss=loss_class+loss_offset

Pthresh = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.999)
Rthresh = (0.5, 0.3)

# ====================================================================================================
training = 0  # ????========================================================================================
# ====================================================================================================
trainer = create_supervised_trainer(net, optimizer, loss)