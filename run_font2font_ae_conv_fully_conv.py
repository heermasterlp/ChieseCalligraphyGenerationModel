# coding: utf-8
import argparse
import tensorflow as tf
from datetime import datetime

from models.autoencoder.model_ae_conv_fully_conv import Font2FontAutoEncoder


def train_model():
    """
    Train auto-encoder model
    :return:
    """
    print("======== Begin to train auto-encoder model =========")
    start_time = datetime.now()
    print("Begin time: {}".format(start_time))
    print("Args: {}".format(args))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print("------- Init model ------")
        model = Font2FontAutoEncoder(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                                     input_width=args.image_size, output_width=args.image_size,
                                     Loss_penalty=args.Loss_penalty, network_dim=args.network_dim)
        model.register_session(sess)
        model.build_model(keep_prob=args.keep_prob, is_training=True)
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume, schedule=args.schedule,
                    freeze_encoder=args.freeze_encoder, sample_steps=args.sample_steps,
                    checkpoint_steps=args.checkpoint_steps)

    end_time = datetime.now()
    print("End time: {}".format(end_time))
    duration = end_time - start_time
    print("Duration hours: {}".format(duration.total_seconds() / 3600.))

    print("======== End train auto-encoder model =======")


def infer_model():
    """
    Infer the auto-encoder model.
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = Font2FontAutoEncoder(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                                     input_width=args.image_size, output_width=args.image_size,
                                     Loss_penalty=args.Loss_penalty, network_dim=args.network_dim)
        model.register_session(sess)
        model.build_model(keep_prob=args.keep_prob, is_training=False)
        model.infer(args.source_obj, args.model_dir, args.save_dir)


parser = argparse.ArgumentParser(description='Train or infer with Auto-encoder model')
parser.add_argument('--mode', dest='mode', required=True, default='train', help='mode: train or infer')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--network_dim', dest='network_dim', type=int, default=64,
                    help="network dim number")
parser.add_argument('--Loss_penalty', dest='Loss_penalty', type=float, default=100.0, help='weight for  loss')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=1.0, help='dropout keep_prob')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=int, default=1, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                    help="freeze encoder weights during training")
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=500,
                    help='number of batches in between two checkpoints')
parser.add_argument('--source_obj', dest='source_obj', help='source obj dataset')
parser.add_argument('--model_dir', dest='model_dir', help='model directory')
parser.add_argument('--save_dir', dest='save_dir', help='save directory')

args = parser.parse_args()


if __name__ == '__main__':
    if args.mode == "train":
        train_model()

    elif args.mode == "infer":
        infer_model()