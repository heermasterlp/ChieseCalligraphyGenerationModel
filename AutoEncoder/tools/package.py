# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import glob
import os
import pickle
import random

font = "qigong"


def package_images():
    save_path = os.path.join("../../../../Data/CalligraphyGenerationModel/AutoEncoder", font)

    train_path = os.path.join(save_path, 'train')
    val_path = os.path.join(save_path, 'val')
    test_path = os.path.join(save_path, 'test')

    train_img_paths = glob.glob(os.path.join(train_path, "*.png"))
    with open(os.path.join(save_path, "train.obj"), "wb") as fout:
        for p in train_img_paths:
            with open(p, 'rb') as fin:
                img_bytes = fin.read()
                pickle.dump(img_bytes, fout)
    print('Process train image end!')


    val_img_paths = glob.glob(os.path.join(val_path, "*.png"))
    with open(os.path.join(save_path, "val.obj"), "wb") as fout:
        for p in val_img_paths:
            with open(p, 'rb') as fin:
                img_bytes = fin.read()
                pickle.dump(img_bytes, fout)
    print('Process val image end!')


    test_img_paths = glob.glob(os.path.join(test_path, "*.png"))
    with open(os.path.join(save_path, "test.obj"), "wb") as fout:
        for p in test_img_paths:
            with open(p, "rb") as fin:
                img_bytes = fin.read()
                pickle.dump(img_bytes, fout)
    print("process test images end!")











if __name__ == '__main__':
    package_images()