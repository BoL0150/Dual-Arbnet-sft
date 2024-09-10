import os
import scipy.io as sio
from os.path import splitext
from tqdm import tqdm
import argparse
import concurrent
import matplotlib.pyplot as plt
# %%
path = "/home/libo/Dual-ArbNet/demodata/mattest/IXI002-Guys-0828-T2-101.mat"
mat = sio.loadmat(path)['img']
print(mat.shape)

fig = plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
plt.imshow(mat,cmap='gray')

plt.show()
# %%
# %%
path = "/home/libo/Dual-ArbNet/demodata/ref_mat/IXI002-Guys-0828-PD-000.mat"
mat = sio.loadmat(path)['img']
print(mat.shape)

fig = plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
plt.imshow(mat,cmap='gray')

plt.show()
# %%
