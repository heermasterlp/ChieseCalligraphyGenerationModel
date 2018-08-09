# coding: utf-8
import tensorflow as tf
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Inference the autoencoder with t-sne algorithm')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')

def main(_):
    pass


if __name__ == '__main__':
    main()