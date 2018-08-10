# coding: utf-8
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple
import pickle

from models.autoencoder.ops import conv2d, deconv2d, lrelu, fc, batch_norm
from models.autoencoder.dataset import TrainDataProvider, InjectDataProvider
from utils.utils import merge, save_concat_images, save_image, scale_back

LossHandle = namedtuple("LossHandle", ["loss"])
InputHandle = namedtuple("InputHandle", ["real_data"])
EvalHandle = namedtuple("EvalHandle", ["network", "target", "source", "code"])
SummaryHandle = namedtuple("SummaryHandle", ["g_merged"])


class Font2FontAutoEncoder(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 network_dim=64, Loss_penalty=100.0, input_filters=1, output_filters=1):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.network_dim = network_dim
        self.Loss_penalty = Loss_penalty
        self.input_filters = input_filters
        self.output_filters = output_filters

        # init all the directories
        self.sess = None

        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")

    def network(self, images, is_training, keep_prob=1.0, reuse=False):
        """
        Network architecture.
        :param images:
        :param is_training:
        :param reuse:
        :return:
        """
        # Encoder conv layers
        # (256, 256) -> (128, 128)
        e1 = conv2d(images, self.network_dim, scope="g_e1_conv")
        e1 = tf.nn.relu(e1)
        e1 = tf.nn.dropout(e1, keep_prob=keep_prob)

        # (128, 128) -> (64, 64)
        e2 = conv2d(e1, self.network_dim * 2, scope="g_e2_conv")
        e2 = batch_norm(e2, is_training, scope="g_e2_bn")
        e2 = tf.nn.relu(e2)
        e2 = tf.nn.dropout(e2, keep_prob=keep_prob)

        # (64, 64) -> (32, 32)
        e3 = conv2d(e2, self.network_dim * 4, scope="g_e3_conv")
        e3 = batch_norm(e3, is_training, scope="g_e3_bn")
        e3 = tf.nn.relu(e3)
        e3 = tf.nn.dropout(e3, keep_prob=keep_prob)

        # (32, 32) -> (16, 16)
        e4 = conv2d(e3, self.network_dim * 8, scope="g_e4_conv")
        e4 = batch_norm(e4, is_training, scope="g_e4_bn")
        e4 = tf.nn.relu(e4)
        e4 = tf.nn.dropout(e4, keep_prob=keep_prob)

        # (16, 16) -> (8, 8)
        e5 = conv2d(e4, self.network_dim * 8, scope="g_e5_conv")
        e5 = batch_norm(e5, is_training, scope="g_e5_bn")
        e5 = tf.nn.relu(e5)
        e5 = tf.nn.dropout(e5, keep_prob=keep_prob)

        # flatten   （8， 8， 64 * 8）-> (8 * 8 * 64 * 8)
        flat1 = tf.layers.flatten(e5)

        # fully connect 1: (8 * 8 * 64 * 8) -> 10000
        fc1 = fc(flat1, 10000, scope="fc1")

        # fully connect 2: 10000 -> 100
        code = fc(fc1, 100, scope="code")

        # fully connect 3: 100 -> 10000
        fc2 = fc(code, 10000, scope="fc2")

        # fully connnect4: 10000 -> (8 * 8 * 64 * 8)
        flat2 = fc(fc2, 8 * 8 * 64 * 8)
        # reshape tensor : [batch_size, 8 * 8 * 64 * 8] -> [batch_size, 8, 8, 64 * 8]
        flat2 = tf.reshape(flat2, [-1, 8, 8, 64 * 8])

        # Decoder
        # (8, 8) -> (16, 16)
        d1 = tf.nn.relu(flat2)
        d1 = deconv2d(d1, [self.batch_size, 16, 16, self.network_dim * 8], scope="g_d1_deconv")
        d1 = batch_norm(d1, is_training, scope="g_d1_bn")
        d1 = tf.nn.dropout(d1, keep_prob=keep_prob)

        # (16, 16) -> (32, 32)
        d2 = tf.nn.relu(d1)
        d2 = deconv2d(d2, [self.batch_size, 32, 32, self.network_dim * 8], scope="g_d2_deconv")
        d2 = batch_norm(d2, is_training, scope="g_d2_bn")
        d2 = tf.nn.dropout(d2, keep_prob=keep_prob)

        # (32, 32) -> (64, 64)
        d3 = tf.nn.relu(d2)
        d3 = deconv2d(d3, [self.batch_size, 64, 64, self.network_dim * 4], scope="g_d3_deconv")
        d3 = batch_norm(d3, is_training, scope="g_d3_bn")
        d3 = tf.nn.dropout(d3, keep_prob=keep_prob)

        # (64, 64) -> (128, 128)
        d4 = tf.nn.relu(d3)
        d4 = deconv2d(d4, [self.batch_size, 128, 128, self.network_dim * 2], scope="g_d4_deconv")
        d4 = batch_norm(d4, is_training, scope="g_d4_bn")
        d4 = tf.nn.dropout(d4, keep_prob=keep_prob)

        # (128, 128) -> (256, 256)
        d5 = tf.nn.relu(d4)
        d5 = deconv2d(d5, [self.batch_size, 256, 256, 1], scope="g_d5_deconv")
        # no batch norm here
        d5 = tf.nn.dropout(d5, keep_prob=keep_prob)

        # output
        output = tf.nn.tanh(d5)

        return output, d5, code

    def build_model(self, keep_prob=1.0, is_training=True):
        """
        Build model of autoencoder.
        :param is_training:
        :return:
        """
        real_data = tf.placeholder(tf.float32, [self.batch_size, self.input_width, self.input_width,
                                                self.input_filters], name="real_A_image")
        # target images
        real_B = real_data[:, :, :, :self.input_filters]

        # source images
        real_A = real_data[:, :, :, :self.input_filters]

        # fake B
        fake_B, fake_B_logits, code = self.network(real_A, keep_prob=keep_prob, is_training=is_training)

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(tf.subtract(fake_B, real_B)))

        # l2 loss
        l2_loss = tf.reduce_mean(tf.square(tf.subtract(fake_B, real_B)))

        # reconstruct loss
        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_B, logits=fake_B_logits))

        # loss
        alpha = 0.6
        loss = alpha * l2_loss + (1 - alpha) * l1_loss

        # summaries
        loss_summary = tf.summary.scalar("loss", loss)
        g_merged_summary = tf.summary.merge([loss_summary])

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data)
        loss_handle = LossHandle(loss=loss)
        eval_handle = EvalHandle(network=fake_B, target=real_B, source=real_A, code=code)
        summary_handle = SummaryHandle(g_merged=g_merged_summary)

        # those operations will be shared make them visiual globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        """
        Register tf session.
        :param sess:
        :return:
        """
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        """
        Retrieve trainale vars.
        :param freeze_encoder:
        :return:
        """
        t_vars = tf.trainable_variables()

        g_vars = [var for var in t_vars if "g_" in var.name]

        if freeze_encoder:
            # exclude encoder weights
            print("freeze encoder weights")
            g_vars = [var for var in g_vars if not ("g_e") in var.name]

        return g_vars

    def retrieve_generator_vars(self):
        """

        :return:
        """
        all_vars = tf.global_variables()
        generator_vars = [var for var in all_vars if "g_" in var.name]
        return generator_vars

    def retrieve_handles(self):
        """

        :return:
        """
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")
        return input_handle, loss_handle, eval_handle, summary_handle

    def get_model_id_and_dir(self):
        """

        :return:
        """
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        """

        :param saver:
        :param step:
        :return:
        """
        model_name = "autoencoder.model"
        model_id, model_dir = self.get_model_id_and_dir()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):
        """

        :param saver:
        :param model_dir:
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images):
        """

        :param input_images:
        :return:
        """
        input_handle, loss_handle, eval_handle, summary_handle = self.retrieve_handles()

        fake_images, real_images, loss, code = self.sess.run([eval_handle.network, eval_handle.target,
                                                        loss_handle.loss, eval_handle.code],
                                                       feed_dict={
                                                           input_handle.real_data: input_images
                                                       })
        return fake_images, real_images, loss, code

    def validate_model(self, images, epoch, step):
        """
        Validate this auto-encoder model.
        :param images:
        :param epoch:
        :param step:
        :return:
        """
        fake_images, real_images, loss, code = self.generate_fake_samples(images)
        print("Sample: loss: %.5f " % (loss))

        merged_fake_images = merge(scale_back(fake_images), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_images), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_fake_images, merged_real_images], axis=1)

        model_id, _ = self.get_model_id_and_dir()
        model_sample_dir = os.path.join(self.sample_dir, model_id)
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        sample_img_path = os.path.join(model_sample_dir, "sample_%04d_%06d.png" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair)

    def infer(self, source_obj, model_dir, save_dir):
        """
        Inference this auto-encoder model.
        :param source_obj:
        :param model_dir:
        :param save_dir:
        :return:
        """
        source_provider = InjectDataProvider(source_obj)
        source_iter = source_provider.get_iter(self.batch_size)

        tf.global_variables_initializer().run()

        saver = tf.train.Saver(max_to_keep=100)
        _, model_dir = self.get_model_id_and_dir()
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        code_list = None
        for source_imgs in source_iter:
            fake_imgs, real_imgs, loss, code = self.generate_fake_samples(source_imgs)
            print(code.shape)
            if code_list is None:
                code_list = code.copy()
            else:
                code_list = np.concatenate(code_list, code)

            print(code_list.shape)

            merged_fake_images = merge(fake_imgs, [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)
        # with open(os.path.join(save_dir, "code.txt"), 'w') as f:
        #     for code in code_list:
        #         f.write(code)
        #         f.write("\n")

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, freeze_encoder=False,
              sample_steps=1500, checkpoint_steps=15000):
        """
        Training this auto-encoder model.
        :param lr:
        :param epoch:
        :param schedule:
        :param resume:
        :param freeze_encoder:
        :param sample_steps:
        :param checkpoint_steps:
        :return:
        """
        g_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        input_handle, loss_handle, _, summary_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered!")

        tf.set_random_seed(100)

        print("Prepare training dataset!")
        data_provider = TrainDataProvider(self.data_dir)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val(size=self.batch_size)
        train_sample = data_provider.get_train_sample(size=self.batch_size)

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.loss, var_list=g_vars)

        saver = tf.train.Saver(max_to_keep=100)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        tf.global_variables_initializer().run()
        real_data = input_handle.real_data

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            # update learning rate for lr
            # if (ei+1) % schedule == 0:
            #     update_lr = current_lr / 2.
            #     update_lr = max(update_lr, 0.00002)
            #     print("decay learing rate  from %.6f to %.7f" % (current_lr, update_lr))
            #     current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                batch_images = batch

                _, loss, g_summary = self.sess.run([g_optimizer, loss_handle.loss, summary_handle.g_merged],
                                                   feed_dict={
                                                       real_data: batch_images,
                                                       learning_rate: current_lr
                                                   })
                passed_time = time.time() - start_time

                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, loss: %.5f"
                print(log_format % (ei, bid, total_batches, passed_time, loss))
                summary_writer.add_summary(g_summary, counter)

            # validation in each epoch used the train samples
            # self.validate_model(val_batch_iter, ei, counter)
            self.validate_model(train_sample, ei, counter)

            # save checkpoint in each 50 epoch
            # if (ei + 1) % 50 == 0:
            # self.checkpoint(saver, counter)

        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)

