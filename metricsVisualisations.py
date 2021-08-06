import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import cluster
from scipy.spatial.distance import cdist

non_ref_data = pd.read_csv("data/training/training_metrics.csv")
ref_data = pd.read_csv("data/training/training_37_ref_metrics.csv")

images = non_ref_data['image'].tolist()

brisque = non_ref_data['brisque'].tolist()
sharpness = non_ref_data['pydom_sharpness'].tolist()

#need to save plots

fig, ax = plt.subplots()
ax.scatter(brisque, sharpness)
fig.set_size_inches(12, 8)
for i, txt in enumerate(images):
    ax.annotate(txt, (brisque[i], sharpness[i]))

mse = ref_data['mse'].tolist()
del mse[16]
rmse = ref_data['rmse'].tolist()
del rmse[16]
psnr = ref_data['psnr'].tolist()
del psnr[16]
ssim = ref_data['ssim'].tolist()
del ssim[16]
uqi = ref_data['uqi'].tolist()
del uqi[16]
rand = ref_data['adapted_rand_error'].tolist()
del rand[16]

del images[16]
del brisque[16]

fig, ax = plt.subplots()
ax.scatter(brisque, mse)
fig.set_size_inches(12, 8)
for i, txt in enumerate(images):
    ax.annotate(txt, (brisque[i], mse[i]))

fig, ax = plt.subplots()
ax.scatter(brisque, rmse)
fig.set_size_inches(12, 8)
for i, txt in enumerate(images):
    ax.annotate(txt, (brisque[i], rmse[i]))

fig, ax = plt.subplots()
ax.scatter(brisque, psnr)
fig.set_size_inches(12, 8)
for i, txt in enumerate(images):
    ax.annotate(txt, (brisque[i], psnr[i]))

ssim0 = []
for index in ssim:
    ssim0.append(eval(index)[0])
fig, ax = plt.subplots()
ax.scatter(brisque, ssim0)
fig.set_size_inches(12, 8)
for i, txt in enumerate(images):
    ax.annotate(txt, (brisque[i], ssim0[i]))

fig, ax = plt.subplots()
ax.scatter(brisque, uqi)
fig.set_size_inches(12, 8)
for i, txt in enumerate(images):
    ax.annotate(txt, (brisque[i], uqi[i]))

rand0 = []
for index in rand:
    rand0.append(eval(index)[0])
fig, ax = plt.subplots()
ax.scatter(brisque, rand0)
fig.set_size_inches(12, 8)
for i, txt in enumerate(images):
    ax.annotate(txt, (brisque[i], rand0[i]))
