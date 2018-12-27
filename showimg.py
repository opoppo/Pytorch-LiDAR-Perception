import numpy as np
import torch
import cv2
import math

img = np.load('./testset/img.npy')

for im in img:
    cv2.imshow('scan', im)
    cv2.waitKey()