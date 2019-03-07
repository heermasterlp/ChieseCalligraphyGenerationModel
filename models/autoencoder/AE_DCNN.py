# coding: utf-8
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple

from utils.ops import conv2d, deconv2d, lrelu, fc, batch_norm
from models.autoencoder.dataset import TrainDataProvider, InjectDataProvider
from utils.utils import merge, save_concat_images, save_image, scale_back

LossHandle = namedtuple("LossHandle", ["loss", "l1_loss", "l2_loss"])
InputHandle = namedtuple("InputHandle", ["real_data"])
EvalHandle = namedtuple("EvalHandle", ["network", "target", "source"])
SummaryHandle = namedtuple("SummaryHandle", ["g_merged"])


class AutoEncoderDCNN(object):
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

        self.sess = None

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

    def encoder(self, images, is_training, reuse=False):
        """
        Encoder network
        :param images:
        :param is_training:
        :param reuse:
        :return:
        """
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer, keep_rate=1.0):
                # act = lrelu(x)
                enc = tf.nn.relu(x)
                enc = tf.nn.dropout(enc, keep_rate)
                enc = conv2d(enc, output_filters=output_filters, scope="g_e%d_conv" % layer)

                # batch norm is important for ae, or aw would output nothing!!!
                enc = batch_norm(enc, is_training, scope="g_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.network_dim, scope="g_e1_env")   # 128 x 128
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.network_dim * 2, 2)  # 64 x 64
            e3 = encode_layer(e2, self.network_dim * 4, 3)  # 32 x 32
            e4 = encode_layer(e3, self.network_dim * 8, 4)  # 16 x 16
            e5 = encode_layer(e4, self.network_dim * 8, 5)  # 8 x 8
            e6 = encode_layer(e5, self.network_dim * 8, 6)  # 4 x 4
            e7 = encode_layer(e6, self.network_dim * 8, 7)  # 2 x 2
            e8 = encode_layer(e7, self.network_dim * 8, 8)  # 1 x 1

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, is_training, reuse=False):
        """
        Decoder network
        :param encoded:
        :param encoding_layers:
        :param is_training:
        :param reuse:
        :return:
        """
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), \
                                              int(s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, keep_rate=1.0):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width, output_width, output_filters],
                               scope="g_d%d_deconv" % layer)

                if layer != 8:
                    # normalization for last layer is very important, otherwise GAN is unstable
                    dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                dec = tf.nn.dropout(dec, keep_prob=keep_rate)
                return dec

            d1 = decode_layer(encoded, s128, self.network_dim * 8, layer=1, enc_layer=encoding_layers["e7"])
            d2 = decode_layer(d1, s64, self.network_dim * 8, layer=2, enc_layer=encoding_layers["e6"])
            d3 = decode_layer(d2, s32, self.network_dim * 8, layer=3, enc_layer=encoding_layers["e5"])
            d4 = decode_layer(d3, s16, self.network_dim * 8, layer=4, enc_layer=encoding_layers["e4"])
            d5 = decode_layer(d4, s8, self.network_dim * 4, layer=5, enc_layer=encoding_layers["e3"])
            d6 = decode_layer(d5, s4, self.network_dim * 2, layer=6, enc_layer=encoding_layers["e2"])
            d7 = decode_layer(d6, s2, self.network_dim, layer=7, enc_layer=encoding_layers["e1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None)

            output = tf.nn.tanh(d8)  # (-1, 1)
            return output, d8

    def network(self, images, is_training, reuse=False):
        """
        Network architecture.
        :param images:
        :param is_training:
        :param reuse:
        :return:
        """
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)
        output, d8 = self.decoder(e8, enc_layers, is_training=is_training, reuse=reuse)
        return output, d8

    def build_model(self, is_training=True):
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
        fake_B, fake_B_logits = self.network(real_A, is_training=is_training)

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(tf.subtract(fake_B, real_B)))

        # l2 loss
        l2_loss = tf.reduce_mean(tf.square(tf.subtract(fake_B, real_B)))

        # reconstruct loss
        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_B, logits=fake_B_logits))

        # tv loss
        # width = self.output_width
        # tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :width - 1, :, :]) / width
        #            + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) / width)

        # loss
        alpha = 0.6
        # loss = alpha * l2_loss + (1 - alpha) * l1_loss
        loss = ce_loss

        # summaries
        loss_summary = tf.summary.scalar("loss", loss)
        g_merged_summary = tf.summary.merge([loss_summary])

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data)
        loss_handle = LossHandle(loss=loss, l1_loss=l1_loss, l2_loss=l2_loss)
        eval_handle = EvalHandle(network=fake_B, target=real_B, source=real_A)
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

        fake_images, real_images, loss = self.sess.run([eval_handle.network, eval_handle.target,
                                                        loss_handle.loss],
                                                       feed_dict={
                                                           input_handle.real_data: input_images
                                                       })
        return fake_images, real_images, loss

    def validate_model(self, images, epoch, step):
        """

        :param images:
        :param epoch:
        :param step:
        :return:
        """
        fake_images, real_images, loss = self.generate_fake_samples(images)
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

    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        """

        :param save_dir:
        :param model_dir:
        :param model_name:
        :return:
        """
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, model_dir, save_dir):
        """

        :param source_obj:
        :param model_dir:
        :param save_dir:
        :return:
        """
        source_provider = InjectDataProvider(source_obj)
        source_iter = source_provider.get_iter(self.batch_size)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        for source_imgs in source_iter:
            fake_imgs = self.generate_fake_samples(source_imgs)[0]
            merged_fake_images = merge(fake_imgs, [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, freeze_encoder=False,
              sample_steps=1500, checkpoint_steps=15000):
        """

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

                _, loss, l1_loss, l2_loss, g_summary = self.sess.run([g_optimizer, loss_handle.loss, loss_handle.l1_loss, loss_handle.l2_loss, summary_handle.g_merged],
                                                   feed_dict={
                                                       real_data: batch_images,
                                                       learning_rate: current_lr
                                                   })
                passed_time = time.time() - start_time

                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, loss: %.5f, l1_loss: %.5f, l2_loss: %.5f"
                print(log_format % (ei, bid, total_batches, passed_time, loss, l1_loss, l2_loss))
                summary_writer.add_summary(g_summary, counter)

            # validation in each epoch used the train samples
            self.validate_model(val_batch_iter, ei, counter)
            # self.validate_model(train_sample, ei, counter)

            # save checkpoint in each 50 epoch
            # if (ei + 1) % 50 == 0:
            # self.checkpoint(saver, counter)

        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)




