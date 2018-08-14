# coding: utf-8
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple

from utils.ops import conv2d, deconv2d, lrelu, fc, batch_norm
from utils.dataset import TrainDataProvider, InjectDataProvider
from utils.utils import scale_back, merge, save_concat_images, save_image


LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss", "l1_loss", "tv_loss", "d_loss_real", "d_loss_fake"])
InputHandle = namedtuple("InputHandle", ["real_data"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source"])
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged"])


class Font2FontCGAN(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 generator_dim=64, discriminator_dim=64, L1_penalty=100.0, Lconst_penalty=15.0, Ltv_penalty=0.0,
                 input_filters=1, output_filters=1):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
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

    def encoder(self, images, is_training, reuse=False):
        """

        :param images:
        :param is_training:
        :param reuse:
        :return:
        """
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="g_e1_env")
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            e8 = encode_layer(e7, self.generator_dim * 8, 8)

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, is_training, reuse=False):
        """

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

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width, output_width, output_filters],
                               scope="g_d%d_deconv" % layer)
                if layer != 8:
                    # normalization for last layer is very important, otherwise GAN is unstable
                    dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec
            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e7"],
                              dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["e1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.tanh(d8) # scale to (-1, 1)
            return output

    def generator(self, images, is_training, reuse=False):
        """

        :param images:
        :param is_training:
        :param reuse:
        :return:
        """
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)
        output = self.decoder(e8, enc_layers, is_training=is_training, reuse=reuse)
        return output, e8

    def discriminator(self, image, is_training, reuse=False):
        """

        :param image:
        :param is_training:
        :param reuse:
        :return:
        """
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # [batch, 256, 256, 1] -> [batch, 128, 128, 64]
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))

            # [batch, 128, 128, 64] -> [batch, 64, 64, 64 * 2]
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 4, scope="d_h1_conv"),
                                  is_training, scope="d_bn_1"))

            # [batch, 64, 64, 64 * 2]   ->  [batch, 32, 32, 64 * 4]
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
                                  is_training, scope="d_bn_2"))

            # [batch, 32, 32, 64 * 4]   ->  [batch, 31, 31, 64 * 8]
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
                                  is_training, scope="d_bn_3"))

            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")

            return tf.sigmoid(fc1), fc1

    def build_model(self, is_training=True):
        """

        :param is_training:
        :return:
        """
        real_data = tf.placeholder(tf.float32, [self.batch_size, self.input_width, self.input_width,
                                                self.input_filters + self.output_filters], name="real_A_and_B_image")
        # target images
        real_B = real_data[:, :, :, :self.input_filters]

        # source images
        real_A = real_data[:, :, :, self.input_filters: self.input_filters + self.output_filters]

        # fake B
        fake_B, encoded_real_A = self.generator(real_A, is_training=is_training)

        real_AB = tf.concat([real_A, real_B], 3)
        fake_AB = tf.concat([real_A, fake_B], 3)

        # Note it is not possible to set reuse flag back to False initialized all variables before setting reuse to True
        real_D, real_D_logits = self.discriminator(real_AB, is_training=is_training, reuse=False)
        fake_D, fake_D_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True)

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(fake_B - real_B))

        # l2 loss
        l2_loss = tf.reduce_mean(tf.square(real_B - fake_B))

        theta = 0.6
        l_loss =  theta * l1_loss + (1 - theta) * l2_loss

        # total variation loss
        width = self.output_width
        tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :width - 1, :, :]) / width
                   + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) / width) * self.Ltv_penalty

        # original g loss
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                        labels=tf.ones_like(fake_D)))

        # binary real / fake loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
                                                                             labels=tf.ones_like(real_D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                             labels=tf.zeros_like(fake_D)))

        # D loss
        d_loss = d_loss_real + d_loss_fake

        # G loss
        g_loss = g_loss + tv_loss + l_loss

        # summaries
        l1_loss_summary = tf.summary.scalar("l1_loss", l_loss)
        tv_loss_summary = tf.summary.scalar("tv_loss", tv_loss)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)
        d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_summary = tf.summary.scalar("d_loss", d_loss)
        d_merged_summary = tf.summary.merge([d_loss_summary, d_loss_real_summary, d_loss_fake_summary])
        g_merged_summary = tf.summary.merge([g_loss_summary, tv_loss_summary, l1_loss_summary])

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data)
        loss_handle = LossHandle(d_loss=d_loss, g_loss=g_loss, l1_loss=l_loss, tv_loss=tv_loss,
                                 d_loss_real=d_loss_real, d_loss_fake=d_loss_fake)

        eval_handle = EvalHandle(encoder=encoded_real_A, generator=fake_B, target=real_B, source=real_A)

        summary_handle = SummaryHandle(d_merged=d_merged_summary, g_merged=g_merged_summary)

        # those operations will be shared make them visiual globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        """

        :param sess:
        :return:
        """
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        """

        :param freeze_encoder:
        :return:
        """
        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if "d_" in var.name]
        g_vars = [var for var in t_vars if "g_" in var.name]

        if freeze_encoder:
            # exclude encoder weights
            print("freeze encoder weights")
            g_vars = [var for var in g_vars if not ("g_e") in var.name]

        return g_vars, d_vars

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
        summary_handle = getattr(self,"summary_handle")
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
        model_name = "font2font.model"
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

        fake_images, real_images, \
        d_loss, g_loss, l1_loss, tv_loss = self.sess.run([eval_handle.generator,
                                                          eval_handle.target,
                                                          loss_handle.d_loss,
                                                          loss_handle.g_loss,
                                                          loss_handle.l1_loss,
                                                          loss_handle.tv_loss],
                                                         feed_dict={
                                                             input_handle.real_data: input_images
                                                         })
        return fake_images, real_images, d_loss, g_loss, l1_loss, tv_loss

    def validate_model(self, images, epoch, step):
        """

        :param images:
        :param epoch:
        :param step:
        :return:
        """
        fake_images, real_images, d_loss, g_loss, l1_loss, tv_loss = self.generate_fake_samples(images)
        print("Sample: d_loss: %.5f, g_loss: %.5f, l1_loss: %.5f, tv_loss: %.5f" % (d_loss, g_loss, l1_loss, tv_loss))

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
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def train(self, lr_g=0.0002, lr_d=0.000005, epoch=100, schedule=10, resume=True, freeze_encoder=False,
              sample_steps=1500, checkpoint_steps=15000):
        """

        :param lr_g:
        :param lr_d:
        :param epoch:
        :param schedule:
        :param resume:
        :param freeze_encoder:
        :param sample_steps:
        :param checkpoint_steps:
        :return:
        """
        g_vars, d_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        input_handle, loss_handle, _, summary_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered!")

        tf.set_random_seed(100)

        learning_rate_d = tf.placeholder(tf.float32, name="learning_rate_d")
        learning_rate_g = tf.placeholder(tf.float32, name="learning_rate_g")
        d_optimizer = tf.train.AdamOptimizer(learning_rate_d, beta1=0.5).minimize(loss_handle.d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate_g, beta1=0.5).minimize(loss_handle.g_loss, var_list=g_vars)

        tf.global_variables_initializer().run()
        real_data = input_handle.real_data

        data_provider = TrainDataProvider(self.data_dir)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val(size=self.batch_size)

        saver = tf.train.Saver(max_to_keep=100)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr_g = lr_g
        current_lr_d = lr_d
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            # update learning rate for lr_g and lr_d
            if (ei+1) % schedule == 0:
                update_lr_g = current_lr_g / 2.
                update_lr_d = current_lr_d / 2.

                update_lr_g = max(update_lr_g, 0.00002)
                update_lr_d = max(update_lr_d, 0.0000005)
                print("decay learing rate of g from %.6f to %.7f" % (current_lr_g, update_lr_g))
                print("decay learing rate of d from %.6f to %.7f" % (current_lr_d, update_lr_d))

                current_lr_g = update_lr_g; current_lr_d = update_lr_d

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                batch_images = batch

                # Optimize D
                _, batch_d_loss, batch_d_loss_real, \
                batch_d_loss_fake, d_summary = self.sess.run([d_optimizer, loss_handle.d_loss, loss_handle.d_loss_real,
                                                            loss_handle.d_loss_fake, summary_handle.d_merged],
                                                            feed_dict={
                                                                real_data: batch_images,
                                                                learning_rate_d: current_lr_d
                                                            })
                # Optimize G
                _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                feed_dict={
                                                    real_data: batch_images,
                                                    learning_rate_g: current_lr_g
                                                })

                # magic move to Optimize G again
                # according to https://github.com/carpedm20/DCGAN-tensorflow
                # collect all the losses along the way
                _, batch_g_loss, l1_loss,\
                tv_loss, g_summary = self.sess.run([g_optimizer, loss_handle.g_loss, loss_handle.l1_loss,
                                                    loss_handle.tv_loss, summary_handle.g_merged],
                                                    feed_dict={
                                                        real_data: batch_images,
                                                        learning_rate_g: current_lr_g
                                                    })
                passed_time = time.time() - start_time

                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, " + \
                             "l1_loss: %.5f,tv_loss: %.5f, d_loss_real: %.5f, d_loss_fake: %.5f"
                print(log_format % (ei, bid, total_batches, passed_time, batch_d_loss, batch_g_loss, l1_loss, tv_loss,
                                    batch_d_loss_real, batch_d_loss_fake))
                summary_writer.add_summary(d_summary, counter)
                summary_writer.add_summary(g_summary, counter)

            # validation in each epoch
            self.validate_model(val_batch_iter, ei, counter)

            # save checkpoint in each 50 epoch
            # if (ei + 1) % 50 == 0:
            #     self.checkpoint(saver, counter)

        # save the last checkpoint
        # print("Checkpoint: last checkpoint step %d" % counter)
        # self.checkpoint(saver, counter)