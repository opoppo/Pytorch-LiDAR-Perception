import os
# import pcl
import numpy as np
import torch
import cv2
import math
from bBox_2D import bBox_2D
import json
import random
import shutil

cloudata = np.load('./testset/cloudata.npy')
anndata = np.load('./testset/anndata.npy')
img = []

# ==============================
resolution = 999  # res*res !!!   (224 ResNet  299 Inception  1000 Visualization ONLY)
# ==============================

# Cloud data to images
for i, scan in enumerate(cloudata):
    emptyImage = np.zeros([200, 200, 3], np.uint8)
    for dot in scan:
        if dot[0] < 30 and 100 / 6 > dot[1] > -100 / 6:
            emptyImage[int(dot[0] * 180 / 30 + 20), int(dot[1] * 6 + 100)] = (
                int(255 - math.hypot(dot[0], dot[1]) * 255 / 60), int(255 - (dot[0] * 235 / 30 + 20)),
                int(dot[1] * 75 / 15 + 80))
    outImage = cv2.resize(emptyImage, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    for j, label in enumerate(anndata[i]):
        if label[0] < label[1] and (label[4] == -90 or label[4] == 0 or label[4] == 90 or label[4] == -180):
            box = bBox_2D(label[0], label[1], label[3], label[2], -label[4])  # fix annotations!!!
        else:
            box = bBox_2D(label[1], label[0], label[3], label[2], -label[4])

        # print(box.xc,box.yc)
        if box.xc == 0 and box.yc == 0 and box.length == 0 and box.width == 0:
            anndata[i][j] = [0, 0, 0, 0, 0]  # mark with 0
            continue
        # print(' xc ', box.xc, ' yc ', box.yc, ' l ', box.length, ' w ', box.width)
        box.scale(300 / 50, 100, 20)
        box.scale(resolution / 200, 0, 0)

        anndata[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]

        box.bBoxCalcVertxex()
        # cv2.line(outImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(outImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(outImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        # cv2.line(outImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)
        # print(' xc ',box.xc,' yc ',box.yc,' l ',box.length,' w ',box.width,' a ',box.alpha)
    # cv2.imshow('scan', outImage)
    print(i)
    # k=cv2.waitKey()
    # if k == 27:  # Esc for exiting
    #     cv2.destroyAllWindows()
    #     os._exit(1)

    img.append(outImage)

# Flipping
# augmentimg = []
# for i, im in enumerate(img):
#     imflipped = cv2.flip(im, 1)
#     augmentimg.append(imflipped)
# img = img + augmentimg
# del augmentimg
#
# augmentann = np.zeros(anndata.shape, dtype=np.float)
# for i, scan in enumerate(anndata):
#     for j, label in enumerate(scan):
#         if label[0]==0:
#             continue
#         box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
#         box.flipx(axis=int(resolution / 2))
#         augmentann[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]
# anndata = np.concatenate((anndata, augmentann))
# del augmentann

# Adding noise : rotate, translate(x,y), resize
# print('Adding Noise...')
# augmentann = np.zeros(anndata.shape, dtype=np.float)
# for i, scan in enumerate(anndata):
#     for j, label in enumerate(scan):
#         if label[0]==0:
#             continue
#         noiseratio = ((torch.randn(2)).div_(20)).exp_()
#         noiseoffset = (torch.randn(2))
#         box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
#         box.rotate(noiseratio[0])
#         box.resize(noiseratio[1])
#         box.translate(noiseoffset[0], noiseoffset[1])
#         augmentann[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]
# anndata = np.concatenate((anndata, augmentann))
# del augmentann
# img = img + img
#
ll = len(img)

print(cloudata.shape, '\t', anndata.shape, '\t', ll)

# to COCO json dataset and shuffle and split
ann_json = {}
images = []
annotations = []
categories = []
iminfo = {}
anninfo = {}
catinfo = {}
trainsplit, valsplit, testsplit = int(ll * 0.70), int(ll * (0.70 + 0.15)), ll
overfittest = 60
print(trainsplit, valsplit - trainsplit, testsplit - valsplit)
mwidth, mlength, mrotation, marea = 0, 0, 0, 0

shutil.rmtree('./maskrcnn-benchmark/datasets/coco/val2014')
os.mkdir('./maskrcnn-benchmark/datasets/coco/val2014')
shutil.rmtree('./maskrcnn-benchmark/datasets/coco/train2014')
os.mkdir('./maskrcnn-benchmark/datasets/coco/train2014')
shutil.rmtree('./maskrcnn-benchmark/datasets/coco/test2014')
os.mkdir('./maskrcnn-benchmark/datasets/coco/test2014')
shutil.rmtree('./maskrcnn-benchmark/datasets/coco/overfit2014')
os.mkdir('./maskrcnn-benchmark/datasets/coco/overfit2014')  # renew data space

pixel_mean = np.array([0., 0., 0.])
pixel_std = np.array([0., 0., 0.])
for i, im in enumerate(img):
    cv2.imwrite('./maskrcnn-benchmark/datasets/coco/train2014/im%d.jpg' % i, im)
    pixel_mean += np.array([np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])])
    pixel_std += np.array([np.std(im[:, :, 0]), np.std(im[:, :, 1]), np.std(im[:, :, 2])])
    iminfo = {
        "file_name": "im%d.jpg" % i,
        "height": im.shape[0],
        "width": im.shape[1],
        "id": i
    }
    images.append(iminfo)
print(pixel_mean / ll, '==pixel_mean==',pixel_std/ll,'==pixel_std==')

idcount = 0
for j, ann in enumerate(anndata):
    # np.save('./testset/dataset/ann/ann%d' % j, ann)
    for i, label in enumerate(ann):
        if label[0] == 0:
            continue
        box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
        box.xcyc2topleft()
        anninfo = {
            'segmentation': [],
            'area': box.length * box.width,
            'image_id': j,
            'bbox': [box.xtl, box.ytl, box.width, box.length],
            'rotation': box.alpha,
            'category_id': 1,
            'id': idcount,
            'iscrowd': 0
        }
        annotations.append(anninfo)
        idcount += 1
        mwidth += box.width
        mlength += box.length
        marea += box.length * box.width
        mrotation += box.alpha

catinfo = {
    "supercategory": "none",
    "id": 1,
    "name": "car"}
categories.append(catinfo)

imagetrain = (images.copy())[:trainsplit]
imids = set(im['id'] for im in imagetrain)
annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
             annotations)  # get binary inds and ids of ann according to im
annids.remove(None)
anntrain = []
for ann in annotations:
    if ann['image_id'] in imids:  # two different ids !!!!!!!
        anntrain.append(ann)
trainann_json = {'info': {}, 'images': imagetrain, 'annotations': anntrain, 'categories': categories}
with open("./maskrcnn-benchmark/datasets/coco/annotations/trainann.json", 'w', encoding='utf-8') as json_file:
    json.dump(trainann_json, json_file, ensure_ascii=False)

imageval = (images.copy())[trainsplit:valsplit]
imids = set(im['id'] for im in imageval)
annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
             annotations)  # get binary inds and ids of ann according to im
annids.remove(None)
annval = []
for ann in annotations:
    if ann['image_id'] in imids:  # two different ids !!!!!!!
        annval.append(ann)
valann_json = {'info': {}, 'images': imageval, 'annotations': annval, 'categories': categories}
with open("./maskrcnn-benchmark/datasets/coco/annotations/valann.json", 'w', encoding='utf-8') as json_file:
    json.dump(valann_json, json_file, ensure_ascii=False)

imagetest = (images.copy())[valsplit:]
imids = set(im['id'] for im in imagetest)
annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
             annotations)  # get binary inds and ids of ann according to im
annids.remove(None)
anntest = []
for ann in annotations:
    if ann['image_id'] in imids:  # two different ids !!!!!!!
        anntest.append(ann)
testann_json = {'info': {}, 'images': imagetest, 'annotations': anntest, 'categories': categories}
with open("./maskrcnn-benchmark/datasets/coco/annotations/testann.json", 'w', encoding='utf-8') as json_file:
    json.dump(testann_json, json_file, ensure_ascii=False)

imageoverfit = (images.copy())[:overfittest]
imids = set(im['id'] for im in imageoverfit)
annids = set(ann['id'] if ann['image_id'] in imids else None for ann in
             annotations)  # get binary inds and ids of ann according to im
annids.remove(None)
annoverfit = []
for ann in annotations:
    if ann['image_id'] in imids:  # two different ids !!!!!!!
        annoverfit.append(ann)
overfitann_json = {'info': {}, 'images': imageoverfit, 'annotations': annoverfit, 'categories': categories}
with open("./maskrcnn-benchmark/datasets/coco/annotations/overfit.json", 'w', encoding='utf-8') as json_file:
    json.dump(overfitann_json, json_file, ensure_ascii=False)
#
print(mwidth / idcount, mlength / idcount, marea / idcount, mrotation / idcount)
# # 12.588   5.719   131.970   0.0
#

for im in overfitann_json['images']:
    shutil.copyfile('./maskrcnn-benchmark/datasets/coco/train2014/' + im["file_name"],
                    './maskrcnn-benchmark/datasets/coco/overfit2014/' + im["file_name"])

for im in valann_json['images']:
    shutil.move('./maskrcnn-benchmark/datasets/coco/train2014/' + im["file_name"],
                './maskrcnn-benchmark/datasets/coco/val2014/' + im["file_name"])

for im in testann_json['images']:
    shutil.move('./maskrcnn-benchmark/datasets/coco/train2014/' + im["file_name"],
                './maskrcnn-benchmark/datasets/coco/test2014/' + im["file_name"])
