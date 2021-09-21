import os
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tqdm

import cv2

# Specify HSV color range for detection
lower = (25, 40, 200)
upper = (30, 100, 255)

data_dir = 'data/pufferfish-struggle'
num_images = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
progress = np.zeros(num_images)
print(f'Total number of frames: {num_images:d}')

output_file = 'output.csv'

# Counts number of pixels in a given image that fall within specified HSV range
# Input images must match the filename format 'frameXXXX.png' with XXXX being contiguous values from 0000 to 9999
def count_pixels(idx):
    filename = f'frame{idx:04d}.png'
    path = os.path.join(data_dir, filename)
    im_orig = cv2.imread(path)
    im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)
    im = im_orig[275:395, 1860:1980]

    # Convert to HSV space
    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_im, lower, upper)
    count = mask.sum() / 255.

    return count

# Parallel process to count pixels in indexed images
with mp.Pool(mp.cpu_count()) as pool:
    progress = list(tqdm.tqdm(pool.imap(count_pixels, range(num_images)), total=num_images))

# Save data
df = pd.DataFrame(progress)
df.to_csv(output_file)

# Find frames meeting conditions
max = np.max(progress)

frame_0 = np.min([i for i in range(len(progress)) if (progress[i] > 0 and progress[min(i+4, len(progress))] > 0)])
frame_25 = np.min([i for i in range(len(progress)) if progress[i] > max/4.])
frame_50 = np.min([i for i in range(len(progress)) if progress[i] > max/2.])
frame_75 = np.min([i for i in range(len(progress)) if progress[i] > max*3/4.])
frame_100 = np.min([i for i in range(len(progress)) if progress[i] == max])

print(f'Progress 0% at frame: {frame_0:d}')
print(f'Progress 25% at frame: {frame_25:d}')
print(f'Progress 50% at frame: {frame_50:d}')
print(f'Progress 75% at frame: {frame_75:d}')
print(f'Progress 100% at frame: {frame_100:d}')
# print(progress)

# Plot progress
x = np.arange(num_images)
grad = np.gradient(progress)

fig, axis = plt.subplots(1, 2)
axis[0].plot(x, progress)
axis[0].set_title('Progress')
axis[1].plot(x, grad)
axis[1].set_title('Gradient')

plt.show()
