# coding: utf-8

import tensorflow as tf
import numpy as np

from utils.ops import conv2d, deconv2d, lrelu, fc, batch_norm
from models.autoencoder.dataset import TrainDataProvider, InjectDataProvider

data_path = "/Users/liupeng/Documents/PythonProjects/experiments/data/data"

print("Prepare training dataset!")
data_provider = TrainDataProvider(data_path)
total_batches = data_provider.compute_total_batch_num(16)
val_batch_iter = data_provider.get_val(size=16)
train_sample = data_provider.get_train_sample(size=16)

real_data = tf.placeholder(tf.float32, [16, 256, 256, 1], name="real_A_image")
# target images
real_B = real_data[:, :, :, :1]

# source images
real_A = real_data[:, :, :, :1]

tanh = tf.nn.tanh(real_A)

# conv = conv2d(real_A, output_filters=1)


train_batch_iter = data_provider.get_train_iter(16)

with tf.Session() as sess:

    for id, batch in enumerate(train_batch_iter):

        th = sess.run([tanh], feed_dict={real_data: batch})
        print(np.amax(th), np.amin(th))
        break



