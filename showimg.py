import numpy as np
import cv2
from bBox_2D import bBox_2D
import os
import math
import time

cloudata = np.load('./testset/cloudata.npy')
anndata = np.load('./testset/anndata.npy')
img = []

# ==============================
resolution = 999  # res*res !!!   (224 ResNet  299 Inception  1000 Visualization ONLY)
# ------------>   x    annotation box clock wise
# |
# |
# |
# y
# ==============================
keep = [ii for ii in range(len(cloudata))]
discard = []  # index for frames

# Cloud data to images
_pixel_enhance = np.array([-1, 0, 1])
pixel_enhance = np.array([[x, y] for x in _pixel_enhance for y in _pixel_enhance])  # enhance pixel by extra 8
for i, scan in enumerate(cloudata):
    emptyImage = np.zeros([200, 200, 3], np.uint8)
    for dot in scan:
        if dot[0] < 30 and 100 / 6 > dot[1] > -100 / 6:  # in range
            x, y = int(dot[0] * 180 / 30 + 20), int(dot[1] * 6 + 100)
            enhanced = [[x, y] + e for e in pixel_enhance]
            for e in enhanced:
                if e[0] < 200 and 0 <= e[0] and e[1] < 200 and 0 <= e[0]:
                    emptyImage[e[0], e[1]] = (
                        int(255 - math.hypot(dot[0], dot[1]) * 255 / 60), int(255 - (dot[0] * 235 / 30 + 20)),
                        int(dot[1] * 75 / 15 + 80))

    outImage = cv2.resize(emptyImage, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    for j, label in enumerate(anndata[i]):
        if label[4] == -90 or  label[4] == 90 :
            box = bBox_2D(label[1], label[0], label[3], label[2], -label[4])  # fix annotations!!!
        else:
            box = bBox_2D(label[0], label[1], label[3], label[2], -label[4])  # clock wise

        # print(box.xc,box.yc)
        if box.xc == 0 and box.yc == 0 and box.length == 0 and box.width == 0:
            anndata[i][j] = [0, 0, 0, 0, 0]  # mark with 0
            continue
        # print(' xc ', box.xc, ' yc ', box.yc, ' l ', box.length, ' w ', box.width)
        box.scale(300 / 50, 100, 20)
        box.scale(resolution / 200, 0, 0)

        anndata[i][j] = [box.length, box.width, box.xc, box.yc, box.alpha]
        rad = box.alpha * math.pi / 180
        box.bBoxCalcVertxex()
        cv2.line(outImage, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(outImage, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(outImage, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
        cv2.line(outImage, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)
        point = int(box.xc - box.length * 0.8 * np.sin(rad)), int(box.yc + box.length * 0.8 * np.cos(rad))
        cv2.line(outImage, (int(box.xc), int(box.yc)),
                 point,
                 (155, 255, 255), 1, cv2.LINE_AA)

        print(' xc ', box.xc, ' yc ', box.yc, ' l ', box.length, ' w ', box.width, ' a ', box.alpha)
    cv2.imshow('scan', outImage)
    print('frame', i)
    k = cv2.waitKey()
    # print(k,'======current key pressed=======')
    if k == 27:  # Esc for exiting
        cv2.destroyAllWindows()
        os._exit(1)

    if k == 48:  # num 0 to discard this frame
        discard.append(i)
        print('frame', i, 'removed!!')

for ii in discard:
    keep.remove(ii)
cloudata_filtered = np.array(cloudata[keep])
anndata_filtered = np.array(anndata[keep])

np.save('./testset/cloudata%f' % time.time(), cloudata_filtered)
np.save('./testset/anndata%f' % time.time(), anndata_filtered)
