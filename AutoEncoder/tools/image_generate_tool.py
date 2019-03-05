# coding: utf-8
import os
import argparse
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections
import random

font_dir = "../../fontset/qigongscfont.TTF"
font = "qigong"

save_path = "../../../../Data/CalligraphyGenerationModel/AutoEncoder"

char_size = 256


def generate_images():
    font_path = os.path.join(save_path, font)
    if not os.path.exists(font_path):
        os.mkdir(font_path)

    # make train, val and test dir
    train_path = os.path.join(font_path, 'train')
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    val_path = os.path.join(font_path, 'val')
    if not os.path.exists(val_path):
        os.mkdir(val_path)

    test_path = os.path.join(font_path, 'test')
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    # load characters from text file
    train_set = set()
    with open('../../charset/775basic_characters.txt', 'r') as f:
        charset = f.readlines()
        charset = [c.strip() for c in charset]
        train_set = set(charset)
    print('train set len: ', len(train_set))

    all_set = set()
    with open('../../charset/chinese_characters.txt', 'r') as f:
        charset = f.readlines()
        charset = [c.strip() for c in charset]
        all_set = set(charset)
    print('all set len: ', len(all_set))

    # check the intersection of two sets
    diff_set = all_set - train_set
    print('diff set len: ', len(diff_set))

    # generate train set images
    src_font = ImageFont.truetype(font_dir, size=char_size)

    # count = 0
    # for c in train_set:
    #     e = draw_example(c, src_font, image_mode='L', canvas_size=256, x_offset=0, y_offset=0)
    #     if e:
    #         e.save(os.path.join(font_path, 'train', "%s_%04d.png") % (c, count))
    #         count += 1
    #         if count % 100 == 0:
    #             print('Processed %d chars' % count)



    # generate all set images
    # count = 0
    # for c in diff_set:
    #     e = draw_example(c, src_font, image_mode='L', canvas_size=256, x_offset=0, y_offset=0)
    #     if e:
    #         e.save(os.path.join(font_path, 'test', "%s_%04d.png") % (c, count))
    #         count += 1
    #         if count % 100 == 0:
    #             print('Processed %d chars' % count)

    # random select image 200 from testing images
    diff_set = list(diff_set)
    random.shuffle(diff_set)
    random.shuffle(diff_set)
    random.shuffle(diff_set)

    val_set = diff_set[:200]
    print('val set len: ', len(val_set))


    # generate val set images
    # count = 0
    # for c in val_set:
    #     e = draw_example(c, src_font, image_mode='L', canvas_size=256, x_offset=0, y_offset=0)
    #     if e:
    #         e.save(os.path.join(font_path, 'val', "%s_%04d.png") % (c, count))
    #         count += 1
    #         if count % 100 == 0:
    #             print('Processed %d chars' % count)





def draw_single_char(ch, font, image_mode, canvas_size, x_offset, y_offset):
    default_color = 255
    if image_mode == "RGB":
        default_color = (255, 255, 255)
    img = Image.new(image_mode, (canvas_size, canvas_size), default_color)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def draw_example(ch, font, image_mode, canvas_size, x_offset, y_offset):
    dst_img = draw_single_char(ch, font, image_mode, canvas_size, x_offset, y_offset)

    # check the filter example in the hashes or not
    # src_img = draw_single_char(ch, src_font, image_mode, canvas_size, x_offset, y_offset)

    default_color = 255
    if image_mode == "RGB":
        default_color = (255, 255, 255)

    example_img = Image.new(image_mode, (canvas_size, canvas_size), default_color)

    # dst image is left
    example_img.paste(dst_img, (0, 0))
    # src image is right
    # example_img.paste(src_img, (canvas_size, 0))
    return example_img



if __name__ == '__main__':
    generate_images()