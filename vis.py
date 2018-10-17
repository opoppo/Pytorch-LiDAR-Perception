import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lbtoolbox.util as lbu
import torch.utils.data as data
import utils as u
import lbtoolbox.plotting as lbplt
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'E:/Anaconda/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz/'
from visualize import  make_dot

Xtr = np.load("./Xtrt.npy")
ytr_conf = np.load("./ytrt_conf.npy")
ytr_offs = np.load("./ytrt_offs.npy")

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


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

    def forward(self, X):
        y = self.conv1(X)
        return y


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
            nn.Softmax()
        )
        self.OffsetVoteOutput = nn.Sequential(
            nn.Conv1d(256, 2, (1, 3)),
            Reshape(-1, 2)
        )

    def forward(self, X):
        y = self.conv2(X)
        confi = self.ConfidenceOutput(y)
        offset = self.OffsetVoteOutput(y)
        return (confi, offset)


net = Mknet(win_res=48)
optimizer = torch.optim.Adam(net.parameters())
loss_softmax = nn.CrossEntropyLoss()
loss_offset = nn.MSELoss()


class data_set(data.Dataset):
    def __init__(self, DataTensor, TargetTensor1, TargetTensor2):
        self.DataTensor = DataTensor
        self.TargetTensor1 = TargetTensor1
        self.TargetTensor2 = TargetTensor2

    def __getitem__(self, index):
        return self.DataTensor[index], self.TargetTensor1[index], self.TargetTensor2[index]

    def __len__(self):
        return self.DataTensor.size(0)


DataTensor = torch.Tensor(Xtr)
OffsTargetTensor = torch.Tensor(ytr_offs)
ConfTargetTensor = torch.Tensor(ytr_conf)

trainset = data_set(DataTensor, OffsTargetTensor, ConfTargetTensor)

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=512,
    shuffle=True
)


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
        if thresh < pwc + pwa:
            locs.append(u.rphi_to_xy(*win2global(r, phi, dx, dy)))
            probs.append((pwc, pwa))
    return locs, probs


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


def win2global(r, phi, dx, dy):
    y = r + dy
    dphi = np.arctan2(dx, y)  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    return y / np.cos(dphi), phi + dphi


def plot_pr_curve(det, wcs, was, radii=(0.1, 0.3, 0.5, 0.7, 0.9), figsize=(15, 10)):
    R = len(radii)
    assert R == det[0].shape[1], "You forgot to update the radii."

    fig, ax = plt.subplots(figsize=figsize)

    for i, r in enumerate(radii):
        ls = {-1: '--', 0: '-', 1: '-.'}[np.sign(i - R // 2)]
        ax.plot(det[1][:, i], det[0][:, i], label='r={}, all'.format(r), c='#E24A33', ls=ls)
        ax.plot(wcs[1][:, i], wcs[0][:, i], label='r={}, wcs'.format(r), c='#348ABD', ls=ls)
        ax.plot(was[1][:, i], was[0][:, i], label='r={}, was'.format(r), c='#988ED5', ls=ls)

    u.prettify_pr_curve(ax)
    lbplt.fatlegend(ax)
    return fig, ax


# tr_scans = np.load("../tr_sacns.npy")
# tr_wcs = np.load("../tr_wcs.npy")
# tr_was = np.load("../tr_was.npy")
Xva = np.load("./Xva.npy")
# # Xte = np.load( "../Xte.npy" )
va_wcs = np.load("./va_wcs.npy")
# # te_wcs=np.load("../te_wcs.npy")
va_was = np.load("./va_was.npy")
# # te_was=np.load("../te_was.npy")
va_scans = np.load("./va_scans.npy")
#
EPOCH = 1
p1, p2 = 1, 1
for epoch in range(EPOCH):
    for step, (x, yOffs, yConf) in enumerate(train_loader):
        (outConf, outOffs) = net(x)
        g,p=make_dot(outConf),make_dot(outOffs)
        # g.view()
        p.view()
        yConfl = yConf.type(torch.LongTensor)
        # if step == 0:
        #     p1 = 1 / loss_softmax(outConf, yConfl)
        #     p2 = 1 / loss_offset(outOffs, yOffs)
        a,b=loss_softmax(outConf, yConfl),loss_offset(outOffs, yOffs)
        loss =  a+ b*25
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step!=0:
            break
    # print("EPOCH : ",epoch," loss_softmax : ",a," loss_offset : ",b*25," loss_tatal : ", loss)