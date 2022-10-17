import numpy as np
import cv2
import os

from skimage.metrics import structural_similarity
from cpselect_master.cpselect.cpselect import cpselect


def cpselection(img1, img2):
    controlpoints = cpselect(img1,img2)
    cp_img1 = []
    cp_img2 = []
    for controlpoint in controlpoints:
        cp_img1.append([controlpoint['img1_x'], controlpoint['img1_y']])
        cp_img2.append([controlpoint['img2_x'], controlpoint['img2_y']])

    return cp_img1, cp_img2

def alignImages(root):
    n1 = '/frame_00009'
    n2 = '/frame_00009_W'
    img1_path = root + n1 + '.png'
    img2_path = root + n2 + '.png'
    # read the images
    im1 = cv2.imread(img1_path)
    im2 = cv2.imread(img2_path)

    pt1, pt2 = cpselection(img1_path, img2_path)
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)

    # Find homography
    h, mask = cv2.findHomography(pt2, pt1, cv2.RANSAC, 10.)

    # Use homography
    height, width, channels = im2.shape

    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    #if you want to see the result side by side
    #img_con = np.concatenate((im2, im1Reg), axis=1)

    cv2.imwrite("examples/demo1.png", im1Reg)
    print(h)

    return im1Reg, h

def cut_image(dirs, path):
       cut_lx, cut_rx, cut_ty, cut_by = 530, 530, 190, 190
       cut_x, cut_y = 540, 190  # 530, 200

       for j in range(len(dirs)):  # len(d)):
              img_f = (path + "/" + dirs[j])
              img = cv2.imread(img_f)
              print("read: ")
              print(img.shape)

              img = img[:-cut_x, :-cut_y]
              # print(img[cut_y, cut_x:])

              print("write: ")
              print(img.shape)
              # cv2.imwrite(dest_root + "/" + dirs[j], img)

def ssim(img1, img2):
       # Convert images to grayscale
       before_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
       after_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

       before_gray = cv2.resize(before_gray, (2048, 2048))
       after_gray = cv2.resize(after_gray, (2048, 2048))

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

       mask = np.zeros(img1.shape, dtype='uint8')
       filled_after = img2.copy()

       for c in contours:
              area = cv2.contourArea(c)
              if area > 40:
                     x, y, w, h = cv2.boundingRect(c)
                     cv2.rectangle(img1, (x, y), (x + w, y + h), (36, 255, 12), 2)
                     cv2.rectangle(img2, (x, y), (x + w, y + h), (36, 255, 12), 2)
                     cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                     cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

       cv2.imshow('before', img1)
       cv2.imshow('after', img2)
       cv2.imshow('diff', diff)
       cv2.imshow('mask', mask)
       cv2.imshow('filled after', filled_after)
       cv2.waitKey(0)

if __name__ == '__main__':
       alignImages("SkyData Raw Sample")

