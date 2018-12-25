import os
import pcl
import numpy as np
import torch

# class Point(object):
#     def __init__(self,x,y,z):
#         self.x = x
#         self.y = y
#         self.z = z
# points = []

# pathbox = 'D:/JupyterNotebook/testset/_bbox/'
# pathpcd = 'D:/JupyterNotebook/testset/pcd/'
#
# for fpathe, dirs, fs in os.walk(pathbox):
#     cloudata=[]
#     for f in fs:
#         # print(os.path.join(fpathe, f))
#         pcdname=f.replace('.txt','.pcd')
#         print(pcdname)
#         p = pcl.PointCloud()
#         p.from_file(pathpcd+pcdname)
#         parray=p.to_array()
#         cloudata.append(parray)
#
#     # cloudtensor=torch.FloatTensor(cloudata)
#     print(len(cloudata))
#     np.save('D:/JupyterNotebook/testset/cloudata',cloudata)

a=np.load('./testset/cloudata.npy')
b=torch.FloatTensor(a)
print(b.size())


#
# filename = 'D:/1544600733.580758018'
#
# p = pcl.PointCloud()
# p.from_file(filename+'.pcd')
# print(p)
# p.to_file("ppp.txt")
# a=p.to_array()
# print(a)
# with open(filename+'.pcd','r',encoding='UTF-8') as f:
#     for line in  f.readlines()[11:len(f.readlines())-1]:
#         strs = line.split(' ')
#         points.append(Point(strs[0],strs[1],strs[2].strip()))

# fw = open(filename+'.txt','w')
# for i in range(len(points)):
#      linev = points[i].x+" "+points[i].y+" "+points[i].z+"\n"
#      fw.writelines(linev)
# fw.close()
