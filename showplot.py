import visdom
import numpy as np
import torch
import utils as u
import torchvision
from skimage.transform import resize


def plotline(Xtr,win_res):
    Xtr=Xtr.reshape((-1, 450, win_res))
    y=Xtr[1,:,:]
    print(y.shape)
    vis = visdom.Visdom()
    # vis.text('Hello, world!')
    for i,cutout in enumerate(y):
        # print(cutout[::-1])
        x=np.linspace(-1,1,48)
        y=cutout[::-1]
        # pic=np.concatenate((x,y),1)
        # print(pic.shape)
        vis.line(Y=y,X=x)
        if i>=0:
            break

def plotprecrec(result):
    vis = visdom.Visdom()

    for i,line in enumerate(result):
        print(line)
        for p,pr,r,t in enumerate(line):
            X = torch.Tensor()
            # print(pr)
            # for j,prec,rec in enumerate(pr):
            #     X.append(torch.Tensor([prec,rec]))
            # vis.scatter(X)

count=[0,0,0]

def vec2img(vec,res,label):
    img=torch.zeros((1,res,res))
    vec=((vec+1)*res/2-1).type(torch.LongTensor)
    yy=torch.arange(0,len(vec),1).type(torch.LongTensor)

    img[0, yy, vec] = 1
    z=torchvision.transforms.functional.to_tensor(torchvision.transforms.functional.resize(torchvision.transforms.functional.to_pil_image(img), (224, 224)))
    # if label==0:
    #     pass
        # torchvision.utils.save_image(z,"./img/0/none"+str(count[0])+".png")
        # count[0]+=1
    if label==1:
        torchvision.utils.save_image(z, "./img/1/wcs"+str(count[1])+".png")
        count[1] += 1
    elif label==2:
        torchvision.utils.save_image(z, "./img/2/was"+str(count[2])+".png")
        count[2] += 1


# def batchvec2img(batchvec,res):
#     # bimg=torch.zeros((len(batchvec),1,224,224))
#
#     # ii=torch.arange(0,len(batchvec),1).type(torch.LongTensor)
#     # bimg=[vecone(vec,res,bimg,1) for vec in batchvec]
#     # oo=list(vec2img(vec,res) for vec in batchvec]))
#     oo=list(vec2img(vec,res) for vec in batchvec)
#
#     return  torch.stack(oo)   # !!!!!!!!!!!!!!!!!!!!!cuda type

def plotvec2img(Xtr,win_res):
    Xtr=Xtr.reshape((-1, win_res))
    vis = visdom.Visdom()
    for i, vec in enumerate(Xtr):
        if i%(450*35330/100)==0:
            Xtr_image =resize(vec2img(vec, res=win_res),(224,224))
            vis.image(Xtr_image)
        pass




