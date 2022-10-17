"""from skimage.metrics import structural_similarity
import cv2
import numpy as np

before = cv2.imread('/home/aylin/demo/projects/images/frame_000027_W.png')
after = cv2.imread('/home/aylin/demo/projects/images/frame_000027.png')

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = structural_similarity(before_gray, after_gray, full=True)
print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff',diff)
cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)
cv2.waitKey(0)
"""
# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
import cv2
import numpy as np

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = []
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=100, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = []
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299, 299, 3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std


path = "/home/CommonDataSets/SkyDataV2/train/0027_Z"
list = os.listdir(path)

path_T = "/home/CommonDataSets/SkyDataV2/train/0027_Z_T"
list_T = os.listdir(path)

img1 = cv2.imread("/home/CommonDataSets/SkyDataV2/train/0027_Z/frame_000001.png")
img2 = cv2.imread("/home/CommonDataSets/SkyDataV2/train/0027_Z/frame_000002.png")
result = np.stack((img1, img2), axis=0)
print(result.dtype)
print(result.shape)

img1_T = cv2.imread("/home/CommonDataSets/SkyDataV2/train/0027_Z_T/frame_000001.png")
img2_T = cv2.imread("/home/CommonDataSets/SkyDataV2/train/0027_Z_T/frame_000002.png")
result = np.append(result, [img1_T], axis=0)
result = np.append(result, [img2_T], axis=0)

for i in list:
    if (i == list[0]) or (i == list[1]):
        continue

    print("for loop is continue")
    img = cv2.imread(path + "/" + i)
    img_T = cv2.imread(path_T + "/" + i)
    result = np.append(result, [img], axis=0)
    result = np.append(result, [img_T], axis=0)

print(result.dtype)
print(result.shape)

# calculate inception score
is_avg, is_std = calculate_inception_score(result)
print('score', is_avg, is_std)