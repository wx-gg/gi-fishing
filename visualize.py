import numpy as np
import matplotlib.pyplot as plt

import cv2

## Based on code from https://realpython.com/python-opencv-color-spaces/

# Specify HSV color range for detection
lower = (25, 40, 200)
upper = (30, 100, 255)

num_images = 1741
progress = np.zeros(num_images)

# for i in range(num_images):
i = 1000
print(i)
path = f'data/pufferfish-perfect/frame{i:04d}.png'

im_orig = cv2.imread(path)
im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)
im = im_orig[275:395, 1860:1980]

# fig, axis = plt.subplots(2, 3)
# plt.subplot(1, 1)
fig = plt.figure(1)
plt.imshow(im)
# plt.show()

# Visualize in RGB space
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

r, g, b = cv2.split(im)
fig2 = plt.figure(2)
# plt.subplot(1, 2)
axis = fig2.add_subplot(1, 1, 1, projection="3d")

pixel_colors = im.reshape((np.shape(im)[0]*np.shape(im)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
axis.set_title("Pixels in RGB space")
# plt.show()

# Visualize in HSV space
hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_im)
fig3 = plt.figure(3)
axis = fig3.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
axis.set_title("Pixels in HSV space")
# plt.show()

mask = cv2.inRange(hsv_im, lower, upper)
result = cv2.bitwise_and(im, im, mask=mask)

fig4 = plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

print(mask.sum())
print(mask.max())
print(mask.min())
print(mask.sum() / 255.)
