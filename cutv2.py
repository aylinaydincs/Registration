import os.path as osp
import numpy as np
import cv2
import os, sys

if __name__ == '__main__':
    path = "/home/aylin/yolo/Registration/images"
    dest_root = "/home/aylin/yolo/Registration/raw_cut"

    dirs = os.listdir(path)

    cut_lx, cut_rx, cut_ty, cut_by = 530, 530, 190, 190
    cut_x, cut_y = 540, 190  # 530, 200

    for j in range(len(dirs)):  # len(d)):
        img_f = (path + "/" + dirs[j])
        img = cv2.imread(img_f)
        print("read: ")
        print(img.shape)

        img = img[:-cut_x, :-cut_y]
        #print(img[cut_y, cut_x:])

        print("write: ")
        print(img.shape)
        #cv2.imwrite(dest_root + "/" + dirs[j], img)

