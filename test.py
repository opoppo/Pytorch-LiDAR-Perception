import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lbtoolbox.util as lbu
import torch.utils.data as data
import utils as u
import lbtoolbox.plotting as lbplt
import matplotlib.pyplot as plt
from visualize import  make_dot
import  time
import os
import math
import torchvision
# np.set_printoptions(threshold=1000)
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def calcPreRec(_X,_scans,_wcs,_was,_r,ts):
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
    alldets, (wcdets, wadets) = pred2det_comb(_scans, pred_y_conf_batches, pred_y_offs_batches, thresh=ts,out_rphi=False)
    gt_all = [wcs + was for wcs, was in zip(_wcs, _was)]
    precs, recs = u.precrec(alldets, gt_all, _r, pred_rphi=False, gt_rphi=True)
    return precs,recs

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
        if thresh < np.exp(pwc)+np.exp(pwa):
            locs.append(u.rphi_to_xy(*win2global(r, phi, dx, dy)))
            probs.append((np.exp(pwc),np.exp(pwa)))
    return locs, probs


def win2global(r, phi, dx, dy):
    y = r + dy
    dphi = np.arctan2(dx, y)  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    return y / np.cos(dphi), phi + dphi

def compute_precrecs(
    scans, pred_conf, pred_offs, gt_wcs, gt_was,
    ts=(1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1-1e-3, 1-1e-5),
    rs=(0.1, 0.3, 0.5, 0.7, 0.9)
):
    gt_all = [wcs+was for wcs, was in zip(gt_wcs, gt_was)]

    if isinstance(rs, (float, int)):
        rs = (rs,)

    mkgt = lambda: np.full((len(ts), len(rs)), np.nan)
    precs   , recs    = mkgt(), mkgt()
    precs_wc, recs_wc = mkgt(), mkgt()
    precs_wa, recs_wa = mkgt(), mkgt()

    for i, t in enumerate(ts):
        alldets, (wcdets, wadets) = pred2det_comb(scans, pred_conf, pred_offs, thresh=t, out_rphi=False)
        for j, r in enumerate(rs):
            precs[i,j], recs[i,j] = u.precrec(alldets, gt_all, r, pred_rphi=False, gt_rphi=True)
            precs_wc[i,j], recs_wc[i,j] = u.precrec(wcdets, gt_wcs, r, pred_rphi=False, gt_rphi=True)
            precs_wa[i,j], recs_wa[i,j] = u.precrec(wadets, gt_was, r, pred_rphi=False, gt_rphi=True)
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
        y = self.conv1(_x)
        return y


class Mknet(nn.Module):
    def __init__(self, win_res):
        super(Mknet, self).__init__()
        self.conv2 = nn.Sequential(
            Reshape(-1, 1, 1, win_res),
            Slot(1, 64, 5),
            # nn.Dropout2d(0.25),
            Slot(64, 64, 5),
            nn.MaxPool2d((1, 2)),
            # nn.Dropout2d(0.25),
            Slot(64, 128, 5),
            # nn.Dropout2d(0.25),
            Slot(128, 128, 3),
            nn.MaxPool2d((1, 2)),
            # nn.Dropout2d(0.25),
            Slot(128, 256, 5),
            # nn.Dropout2d(0.25)
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

    def forward(self, X):
        y = self.conv2(X)
        confi = self.ConfidenceOutput(y)
        offset = self.OffsetVoteOutput(y)
        return (confi, offset)

#Loading
# Xtr = np.load("./Xtr.npy")
Xtr=torch.load('Xtrt.pt')
ytr_conf = torch.load('ytrt_conf.pt')
ytr_offs = torch.load('ytrt_offs.pt')
Xva = torch.load('Xva.pt')
Xte = torch.load('Xte.pt')

va_wcs = torch.load('va_wcs.pt')
te_wcs=torch.load('te_wcs.pt')
tr_wcs=torch.load('tr_wcs.pt')

va_was = torch.load('va_was.pt')
te_was=torch.load('te_was.pt')
tr_was=torch.load('tr_was.pt')

va_scans = torch.load('va_scans.pt')
te_scans=torch.load('te_scans.pt')
tr_scans=torch.load('tr_scans.pt')

#Setting up
DataTensor = torch.cuda.FloatTensor(Xtr)
OffsTargetTensor = torch.cuda.FloatTensor(ytr_offs)
ConfTargetTensor = torch.cuda.FloatTensor(ytr_conf)
del ytr_conf,ytr_offs,Xtr

trainset = data.TensorDataset(DataTensor, OffsTargetTensor, ConfTargetTensor)

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=60000, #120000 for 4 GPUs ; 80000 for 3 GPUs
    shuffle=True,
    drop_last=False,
    # pin_memory=True,
    # num_workers=12
)
TestTensor = torch.cuda.FloatTensor(Xte)
del Xte
testset = data.TensorDataset(TestTensor)

test_loader=data.DataLoader(
    dataset=testset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    # pin_memory=True
)

net = Mknet(win_res=48)#=====================================

# net=torchvision.models.resnet50(pretrained=True)
# net.conv1=(3,)

net=torch.nn.DataParallel(net.cuda(),device_ids=[0,1])
# net=torch.load('net-all-120-v0.2')
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
loss_class = nn.NLLLoss(torch.Tensor((0.5,10,10)).cuda(non_blocking=True))
loss_offset = nn.MSELoss(reduction='sum')

writer = SummaryWriter()
x = torch.FloatTensor([100])
y = torch.FloatTensor([500])


tranning=1 # ????========================================================================================
#Trainning
if(tranning==1):
    net.train()
    EPOCH = 100
    # p1, p2 = 1, 1
    totaltime,losslist,precslist=[],[],[]
    precs0, recs0,i = 0, 0,0
    for epoch in range(EPOCH):
        time_start = time.time()
        for step, (x, yOffs, yConf) in enumerate(train_loader):
            (outConf, outOffs) = net(x)
            del x
           #torch.Size([15360, 3]) torch.Size([15360, 2])
            yConfl = yConf.type(torch.cuda.LongTensor)

            tgt_noise = ((torch.randn(*yOffs.shape)).div_(20)).exp_().type(torch.cuda.FloatTensor)
            mask = ((yConf != 0).view((-1, 1))).type(torch.cuda.FloatTensor)  # Tensor Type match!!!!
            del yConf
            n = mask.sum()

            a=loss_class(outConf.mul_((1-outConf.exp()).pow_(2)), yConfl)  #Focal loss
            if n > 0:
                b=((loss_offset(outOffs, yOffs.mul_(tgt_noise))).div_(n)).sqrt_()  #RMSE loss
            else:
                b=0
            del tgt_noise,mask,n,yConfl,yOffs
            # print(outConf.shape, yConfl.shape,mask.shape) #torch.Size([15360, 3]) torch.Size([15360]) which is correct
            loss=a+b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_end = time.time()
        totaltime.append(time_end-time_start)
        losslist.append((epoch,a.item(),b.item(),loss.item()))
        print("EPOCH",epoch,"  loss_softmax: %.4f"%a.item(),"  loss_offset: %.4f"%b.item(),"  loss_total: %.4f"%loss.item(),"  epoch_time: %.2f"%(time_end-time_start),"s   estimated_time: %.2f"%((EPOCH-epoch-1)*sum(totaltime)/((epoch+1)*60)),"min")

        # writer.add_histogram('zz/x', x, epoch)
        # writer.add_histogram('zz/y', y, epoch)
        # writer.add_scalar('data/x', x, epoch)
        # writer.add_scalar('data/y', y, epoch)
        # writer.add_scalar('data/loss', loss, epoch)
        # writer.add_scalars('data/scalar_group', {'x': x , 'y': y , 'loss': loss}, epoch)
        # writer.add_text('zz/text', 'zz: this is epoch ' + str(epoch), epoch)

        if epoch%5==0:
            precs,recs=calcPreRec(Xva,va_scans,va_wcs,va_was,_r=0.5,ts=0.9)
            print("precision | recall : %.4f" % precs, " | %.4f" % recs, "on validation set")
            precslist.append((precs,recs))
            if ( not math.isnan(precs)) and ( not math.isnan(recs))and precs>precs0 and recs>recs0:
                torch.save(net, "nettmp")
                precs0=precs
                recs0=recs
                i=0
            elif not math.isnan(precs) and not math.isnan(recs):
                i+=1
                if i>=2:
                    with torch.load('nettmp') as net :
                        print("done training")
        # break#===============================

    torch.save(losslist,"losslist.pt")
    torch.save(precslist,"precslist.pt")

    writer.export_scalars_to_json("./test.json")
    writer.close()

   # Predicting
# if (tranning != 1):
#     net=torch.load('net-all-15-v0.1')
#     print("net loaded")

Pthresh=(1e-3,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.99,1-1e-3)
Rthresh=(0.5,0.3)

net.eval()
for j,rs in enumerate(Rthresh):
    result = []
    for i,ts in enumerate(Pthresh):
        precrec = []
        precs, recs = calcPreRec(Xte, te_scans, te_wcs, te_was, _r=rs, ts=ts)
        print("precision | recall : %.4f" % precs, " | %.4f" % recs, "on test set with threshhold ",ts,"r=",rs)
        # precs,recs=torch.Tensor(precs.from_numpy()),torch.Tensor(recs)
        precrec.append([precs,recs])
    result.append([precrec, ts,rs])
torch.save(result,"result.pt")
torch.save(net,"nettt")
print("net saved")


