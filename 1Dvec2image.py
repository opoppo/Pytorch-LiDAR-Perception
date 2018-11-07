import visdom
import numpy as np
import torch
import utils as u
import showplot as sp
import torch.utils.data as data
import sys
import  time
from pympler.asizeof import asizeof
from pympler.tracker import SummaryTracker

tracker = SummaryTracker()

win_res=48
Xtr = np.load("./Xtr.npy")
# Xtr=torch.load('Xtrt.pt')
print(Xtr.shape)
ytr_conf = torch.load('ytr_conf.pt')
ytr_offs = torch.load('ytr_offs.pt')
# print(Xtr.shape)
vis = visdom.Visdom()

class train_data_set(data.Dataset):
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
lenxtr=len(Xtr)
del ytr_conf,ytr_offs,Xtr

trainset = train_data_set(DataTensor, OffsTargetTensor, ConfTargetTensor)

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=20480,
    shuffle=False,
    drop_last=False,

    # num_workers=12
)
totaltime=[]
# num=0
# bimgflag=0
for step, (x, yOffs, yConf) in enumerate(train_loader):
    time_start = time.time()
    # batchimg=sp.batchvec2img(x,win_res)
    for vec,yc,yo in zip(x,yConf,yOffs):
        sp.vec2img(vec, win_res,yc)

    # if step%5==0:
    #     vis.image(z)

    # if bimgflag==0:
    #     bimg=batchimg
    #     bimgflag=1
    # else:
    #     print(tracker)
    #     bimg=torch.cat((bimg,batchimg),0)
    #     bimg.append()
    #     tracker.print_diff()

    # print(".  ",z.shape,"    ",len(bimg),"    ",asizeof(bimg)/(1024*1024),"    ",asizeof(batchimg)/(1024*1024))
    # del z



    # if  (asizeof(bimg)/(1024*1024*1024))>40:#step>=1 and step%100==0:
    #     torch.save(bimg, "./img/bimg%.2f"%num)
    #     print(len(bimg))
    #     num+=1
    #     del bimg
    #     bimgflag=0

        # print("done saving",num)
    time_end = time.time()
    totaltime.append(time_end - time_start)
    print("step", step,"  time: %.2f" % (time_end - time_start),
          "s   estimated_time: %.2f" % ((int(lenxtr/len(x)) - step - 1) * sum(totaltime) / ((step + 1) * 60)), "min" ," steps to go: ",int(lenxtr/len(x))-step-1)
    # break

# num+=1
# torch.save(bimg, "./img/bimg%.2f"%num)
# print(num,"  ",len(bimg))


