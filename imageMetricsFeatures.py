import skimage
import os
from skimage import io
from PIL import Image
import sys
from dom import DOM
import numpy as np
from skimage import img_as_float
import imquality.brisque as brisque
from skimage import metrics
from sewar import full_ref
import csv

io.use_plugin('pil')
iqa = DOM()
training_folder = 'data/training/images'

header = ['id', 'image', 'brisque', 'pydom_sharpness']
with open('data/training/training_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

f = open('data/training/training_metrics.csv', 'a', newline='')
writer = csv.writer(f)
i = 0
for filename in os.listdir(training_folder):
    entry = [i, filename]
    img = io.imread(os.path.join(training_folder, filename))
    entry.append(brisque.score(img_as_float(img)))
    entry.append(iqa.get_sharpness(img))
    print(entry)
    writer.writerow(entry)
    i += 1
    
f.close()

ref_filename = '37_training.tif'
ref_img = io.imread(os.path.join(training_folder, ref_filename))

header = ['id', 'image', 'mse', 'rmse', 'psnr', 'ssim', 'uqi', 'msssim', 'adapted_rand_error']
with open('data/training/training_37_ref_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

f = open('data/training/training_37_ref_metrics.csv', 'a', newline='')
writer = csv.writer(f)
i = 0

for filename in os.listdir(training_folder):
    entry = [i, filename]
    test_img = io.imread(os.path.join(training_folder, filename))
    entry.append(full_ref.mse(ref_img, test_img))
    entry.append(full_ref.rmse(ref_img, test_img))
    entry.append(full_ref.psnr(ref_img, test_img))
    entry.append(full_ref.ssim(ref_img, test_img))
    entry.append(full_ref.uqi(ref_img, test_img))
    entry.append(full_ref.msssim(ref_img, test_img))
    entry.append(metrics.adapted_rand_error(ref_img, test_img))
    print(entry)
    writer.writerow(entry)
    i += 1
    
f.close()
