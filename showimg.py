import numpy as np
import cv2
from bBox_2D import bBox_2D
import torch

img = np.load('./testset/img.npy')
anndata = np.load('./testset/anndatafixed.npy')

# # noise to rotate, translate(x,y), resize
# print('Adding Noise...')
# for i, scan in enumerate(anndata):
#     for j, label in enumerate(scan):
#         noiseratio = ((torch.randn(2)).div_(20)).exp_()
#         noiseoffset = (torch.randn(2))
#         box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
#         box.rotate(noiseratio[0])
#         box.resize(noiseratio[1])
#         box.translate(noiseoffset[0], noiseoffset[1])
#         anndata[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]

for i, im in enumerate(img):
    # emptyImage = cv2.resize(im, (200, 200), interpolation=cv2.INTER_CUBIC)
    emptyImage = cv2.flip(im, 1)
    del im
    for j, label in enumerate(anndata[i]):
        box = bBox_2D(label[0], label[1], label[2], label[3], label[4])
        # box.Scale(299 / 200, 0, 0)  #==========!!!
        box.flipx(axis=112)
        # box.scale(224 / 200, 0, 0)  # ===== !!!
        box.bBoxCalcVertxex()
        cv2.line(emptyImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(emptyImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)
    outImage = cv2.resize(emptyImage, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('scan', outImage)
    k = cv2.waitKey()

    if k == 27:  # Esc for exiting
        cv2.destroyAllWindows()
        break
