from os.path import join as pjoin
from glob import glob
import json
from itertools import chain
from scipy.spatial.distance import cdist

import numpy as np
import  torch

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# Font which got unicode math stuff.
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Much more readable plots
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Much better than plt.subplots()
#from mpl_toolkits.axes_grid1 import ImageGrid

from ipywidgets import interact, IntSlider, FloatSlider

import lbtoolbox.util as lbu
import lbtoolbox.plotting as lbplt
from lbtoolbox.util import batched
# from lbtoolbox.monitor import ReportMonitor
from lbtoolbox.util import Uninterrupt

import utils as u

def linearize(all_seqs, all_scans, all_detseqs, all_wcs, all_was):
    lin_seqs, lin_scans, lin_wcs, lin_was = [], [], [], []
    # Loop through the "sessions" (correspond to files)
    for seqs, scans, detseqs, wcs, was in zip(all_seqs, all_scans, all_detseqs, all_wcs, all_was):
        # Note that sequence IDs may overlap between sessions!
        s2s = dict(zip(seqs, scans))
        # Go over the individual measurements/annotations of a session.
        for ds, wc, wa in zip(detseqs, wcs, was):
            lin_seqs.append(ds)
            lin_scans.append(s2s[ds])
            lin_wcs.append(wc)
            lin_was.append(wa)
    return lin_seqs, lin_scans, lin_wcs, lin_was
#
def closest_detection(scan, dets, radii):
    if len(dets) == 0:
        return np.zeros_like(scan, dtype=int)

    assert len(dets) == len(radii), "Need to give a radius for each detection!"

    scan_xy = np.array(u.scan_to_xy(scan)).T

    # Distance (in x,y space) of each laser-point with each detection.
    dists = cdist(scan_xy, np.array([u.rphi_to_xy(x,y) for x,y in dets]))

    # Subtract the radius from the distances, such that they are <0 if inside, >0 if outside.
    dists -= radii

    # Prepend zeros so that argmin is 0 for everything "outside".
    dists = np.hstack([np.zeros((len(scan),1)), dists])

    # And find out who's closest, including the threshold!
    return np.argmin(dists, axis=1)

def global2win(r, phi, dr, dphi):
    # Convert to relative, angle-aligned x/y coordinate-system.
    dx = np.sin(dphi-phi) * dr
    dy = np.cos(dphi-phi) * dr - r
    return dx, dy

def win2global(r, phi, dx, dy):
    y = r + dy
    dphi = np.arctan2(dx, y)  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    return y/np.cos(dphi), phi + dphi

basedir = labeldir = "./DROW-data/"
train_names = [f[:-4] for f in glob(pjoin(basedir, 'train', '*.csv'))]
val_names = [f[:-4] for f in glob(pjoin(basedir, 'val', '*.csv'))]
te_names = [f[:-4] for f in glob(pjoin(basedir, 'test', '*.csv'))]
# re_names = [f[:-4] for f in glob(pjoin(basedir, 'reha', '*.csv'))]

tr_seqs, tr_scans = zip(*[u.load_scan(f + '.csv') for f in train_names])
va_seqs, va_scans = zip(*[u.load_scan(f + '.csv') for f in val_names])
te_seqs, te_scans = zip(*[u.load_scan(f + '.csv') for f in te_names])
# re_seqs, re_scans = zip(*[u.load_scan(f + '.csv') for f in re_names])

tr_detseqs, tr_wcdets, tr_wadets = zip(*map(u.load_dets, train_names))
va_detseqs, va_wcdets, va_wadets = zip(*map(u.load_dets, val_names))
te_detseqs, te_wcdets, te_wadets = zip(*map(u.load_dets, te_names))
# re_detseqs, re_wcdets, re_wadets = zip(*map(u.load_dets, re_names))

print("Loaded {:6.2f}k train scans, {:5.2f}k labelled".format(sum(map(len, tr_seqs))/1000, sum(map(len, tr_detseqs))/1000))
print("Loaded {:6.2f}k valid scans, {:5.2f}k labelled".format(sum(map(len, va_seqs))/1000, sum(map(len, va_detseqs))/1000))
print("Loaded {:6.2f}k test  scans, {:5.2f}k labelled".format(sum(map(len, te_seqs))/1000, sum(map(len, te_detseqs))/1000))
# print("Loaded {:6.2f}k reha  scans, {:5.2f}k labelled".format(sum(map(len, re_seqs))/1000, sum(map(len, re_detseqs))/1000))

tr_seqs, tr_scans, tr_wcs, tr_was = linearize(tr_seqs, tr_scans, tr_detseqs, tr_wcdets, tr_wadets)
va_seqs, va_scans, va_wcs, va_was = linearize(va_seqs, va_scans, va_detseqs, va_wcdets, va_wadets)
te_seqs, te_scans, te_wcs, te_was = linearize(te_seqs, te_scans, te_detseqs, te_wcdets, te_wadets)
# re_seqs, re_scans, re_wcs, re_was = linearize(re_seqs, re_scans, re_detseqs, re_wcdets, re_wadets)

tr_seqs = np.array(tr_seqs + tr_seqs)
tr_scans = np.array(tr_scans + [scan[::-1] for scan in tr_scans], dtype=np.float64)

tr_wcs = tr_wcs + [[[d[0], -d[1]] for d in dets] for dets in tr_wcs]
tr_was = tr_was + [[[d[0], -d[1]] for d in dets] for dets in tr_was]

va_scans = np.array(va_scans, dtype=np.float64)
te_scans = np.array(te_scans, dtype=np.float64)
#re_scans = np.array(re_scans, dtype=np.float64)

win_res = 48

Xtr = np.empty((len(tr_scans), 450, win_res), dtype=np.float64)
Xva = np.empty((len(va_scans), 450, win_res), dtype=np.float64)
Xte = np.empty((len(te_scans), 450, win_res), dtype=np.float64)

for i, scan in enumerate(tr_scans):
    Xtr[i] = u.generate_cut_outs(scan, npts=win_res)
for i, scan in enumerate(va_scans):
    Xva[i] = u.generate_cut_outs(scan, npts=win_res)
for i, scan in enumerate(te_scans):
    Xte[i] = u.generate_cut_outs(scan, npts=win_res)
#for i, scan in enumerate(re_scans):
  #  Xre[i] = u.generate_cut_outs(scan, npts=win_res)

Xtr = Xtr.reshape((-1, win_res))
Xtr.shape, Xva.shape, Xte.shape#, Xre.shape  # Keep test-time ones nicely formed.

def generate_votes(scan, wcs, was, rwc=0.6, rwa=0.4):
    N = len(scan)
    y_conf = np.zeros( N, dtype=np.float64)
    y_offs = np.zeros((N, 2), dtype=np.float64)

    alldets = list(wcs) + list(was)
    radii = [0.6]*len(wcs) + [0.4]*len(was)
    dets = closest_detection(scan, alldets, radii)
    labels = [0] + [1]*len(wcs) + [2]*len(was)

    for i, (r, phi) in enumerate(zip(scan, u.laser_angles(N))):
        if 0 < dets[i]:
            y_conf[i] = labels[dets[i]]
            y_offs[i,:] = global2win(r, phi, *alldets[dets[i]-1])

    return y_conf, y_offs

ytr_conf, ytr_offs = map(np.concatenate, zip(*list(map(generate_votes, tr_scans, tr_wcs, tr_was))))
print(".")
torch.save(va_wcs,"va_wcs.pt")
torch.save(te_wcs,"te_wcs.pt")
torch.save(va_was,"va_was.pt")
torch.save(te_was,"te_was.pt")
torch.save(tr_wcs,"tr_wcs.pt")
torch.save(tr_was,"tr_was.pt")
print(".")
torch.save(Xva,"Xva.pt")
torch.save(Xte,"Xte.pt")
torch.save(va_scans,"va_scans.pt")
torch.save(te_scans,"te_scans.pt")
torch.save(tr_scans,"tr_scans.pt")
print(".")
np.save("Xtr.npy",Xtr)
torch.save(ytr_conf,"ytr_conf.pt")
torch.save(ytr_offs,"ytr_offs.pt")
print(".")
torch.save(Xtr[-2048000:],"Xtrt.pt")
torch.save(ytr_conf[-2048000:],"ytrt_conf.pt")
torch.save(ytr_offs[-2048000:],"ytrt_offs.pt")

yva_conf, yva_offs = map(np.concatenate, zip(*list(map(generate_votes, va_scans, va_wcs, va_was))))
torch.save(yva_conf,"yva_conf.pt")
torch.save(yva_offs,"yva_offs.pt")


