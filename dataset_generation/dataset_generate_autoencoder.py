# coding: utf-8
import argparse
import os
import numpy as np
from collections import namedtuple

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections
import pickle

"""
    This tool is used to generate training set, evaluation set and testing set for Auto-Encoder.
    
"""

Char_dataset = namedtuple("Char_dataset", ["char_trainset", "char_valset", "char_testset"])


def load_charset(char_dir):
    """
    Load charset from char file.
    :param char_dir:
    :return:
    """
    with open(char_dir, 'r') as f:
        charset = f.readlines()
        charset = [char.strip() for char in charset]
        return charset


def char_dataset_generate(charset, trainset_num, valset_num, testset_num, shuffle=True):
    """
    Generate train, val and test charset based on different number.
    :param charset:
    :param trainset_num:
    :param valset_num:
    :param testset_num:
    :param shuffle:
    :return:
    """
    if charset is None:
        return
    if len(charset) < trainset_num + valset_num + testset_num:
        print("train set, val set and test set should not be larger than the charset")
        return
    # shuffle
    if shuffle:
        np.random.shuffle(charset)
        np.random.shuffle(charset)
        np.random.shuffle(charset)

    trainset = charset[:trainset_num]
    valset = charset[trainset_num: trainset_num + valset_num]
    testset = charset[trainset_num + valset_num: trainset_num + valset_num + testset_num]

    char_dataset = Char_dataset(char_trainset=trainset, char_valset=valset, char_testset=testset)

    return char_dataset


def pickle_dataset(dataset, dataset_dir, font, image_mode, canvas_size, x_offset, y_offset, flag="train"):
    """
    Pickle char dataset into binary image obj file.
    :param font:
    :param image_mode:
    :param canvas_size:
    :param x_offset:
    :param y_offset:
    :param flag:
    :param dataset:
    :param dataset_dir:
    :return:
    """
    if dataset is None:
        return
    dataset_path = dataset_dir
    if flag == "train":
        dataset_path += "/train.obj"
    elif flag == "val":
        dataset_path += "/val.obj"
    elif flag == "test":
        dataset_path += "/test.obj"

    if os.path.exists(os.path.join(dataset_dir, 'images')):
        dirs = os.listdir(os.path.join(dataset_dir, 'images'))
        for f in dirs:
            if 'jpg' in f:
                os.remove(os.path.join(dataset_dir, 'images', f))
    else:
        os.makedirs(os.path.join(dataset_dir, 'images'))

    for i in range(len(dataset)):
        print(i)
        char = dataset[i]
        # generate image of char
        default_color = 255
        if image_mode == "RGB":
            default_color = (255, 255, 255)

        img = Image.new(image_mode, (canvas_size, canvas_size), default_color)
        draw = ImageDraw.Draw(img)
        draw.text((x_offset, y_offset), char, 0, font=font)
        img.save((dataset_dir + "/images/" + "%04d.jpg") % (i))

    dirs = os.listdir(os.path.join(dataset_dir, 'images'))

    with open(dataset_path, "wb") as fd:

        for f in dirs:
            if 'jpg' not in f:
                continue
            # read image file
            with open(os.path.join(dataset_dir, "images", f), "rb") as f:
                img_bytes = f.read()
                pickle.dump(img_bytes, fd)


parser = argparse.ArgumentParser(description='Dataset generation')

parser.add_argument('--charset_dir', dest='charset_dir', help='charset dir')
parser.add_argument('--data_dir', dest='data_dir', help='data save dir')
parser.add_argument('--trainset_num', dest='trainset_num', type=int, default=500, help="number of traning set")
parser.add_argument('--valset_num', dest='valset_num', type=int, default=500, help='number of val set')
parser.add_argument('--testset_num', dest='testset_num', type=int, default=500, help='number of testing set')
parser.add_argument('--shuffle', dest='shuffle', type=bool, default=True, help='shuffle a charset before processings')

parser.add_argument('--font', dest='font', required=True, help='path of the source font')
parser.add_argument('--char_size', dest='char_size', type=int, default=256, help='character size')
parser.add_argument('--image_mode', dest='image_mode', type=str, default="L",
                    help='image mode, L is grayscale and RGB is rgb image ')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=0, help='y_offset')
# parser.add_argument('--flag', dest='flag', type=str, default='train', help='flag of trian, val or test')


args = parser.parse_args()

if __name__ == '__main__':

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    charset = load_charset(args.charset_dir)
    print("char set len: {}".format(len(charset)))

    char_dataset = char_dataset_generate(charset, args.trainset_num, args.valset_num, args.testset_num)

    if char_dataset:
        print("train set num: {}, val set num: {}, test set num: {}".format(len(char_dataset.char_trainset),
                                                                        len(char_dataset.char_valset),
                                                                        len(char_dataset.char_testset)))
    else:
        print("char dataset should no be None!")
        exit()

    # generate obj file
    # trian set
    font = ImageFont.truetype(args.font, size=args.char_size)
    pickle_dataset(dataset=char_dataset.char_trainset, dataset_dir=args.data_dir, font=font,
                   image_mode=args.image_mode, canvas_size=args.canvas_size, x_offset=args.x_offset,
                   y_offset=args.y_offset, flag="train")

    # val
    pickle_dataset(dataset=char_dataset.char_valset, dataset_dir=args.data_dir, font=font,
                   image_mode=args.image_mode, canvas_size=args.canvas_size, x_offset=args.x_offset,
                   y_offset=args.y_offset, flag="val")

    # test
    pickle_dataset(dataset=char_dataset.char_testset, dataset_dir=args.data_dir, font=font,
                   image_mode=args.image_mode, canvas_size=args.canvas_size, x_offset=args.x_offset,
                   y_offset=args.y_offset, flag="test")

