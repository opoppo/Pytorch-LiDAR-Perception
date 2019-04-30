import os
import numpy as np
import matplotlib.pyplot as plt

pathbox = 'D:/JupyterNotebook/testset/_bbox/'
pathpcd = 'D:/JupyterNotebook/testset/pcd/'

bboxarray = []
count = 0

for fpathe, dirs, fs in os.walk(pathbox):
    for f in fs:
        with open(pathbox + f, 'r')as ff:
            for i, line in enumerate(ff):
                bboxlabel = line.strip().split()
                bboxlabel = np.delete(bboxlabel, (0, 1, 4, 7, 8), axis=0)
                bboxlabel = list(map(float, bboxlabel))
                bboxarray.append(bboxlabel)
                count += 1
                print(count)

    # annotationdata = np.array(annotationdata, dtype=np.float)
    # print(annotationdata.shape)
bboxarray = np.array(bboxarray)
y = bboxarray[:, 3]
x = bboxarray[:, 2]
a = bboxarray[:, 4].astype(int) + 180
b = np.bincount(a)
bx = np.array(range(-180, 181))

# plt.quiver(y, x,5*np.sin(a),5*np.cos(a) ,color='#9999ff')

plt.plot(y, x, '+')
plt.scatter(0, 0, c='orange', marker='^')
plt.show()
plt.bar(bx, b, facecolor='#9999ff', edgecolor='#9999ff')
# plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
plt.show()
