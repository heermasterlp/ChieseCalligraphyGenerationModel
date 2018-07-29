# coding: utf-8
import argparse
import tensorflow as tf
from datetime import datetime

from models.cgan.model_cgan import Font2FontCGAN


def train_model():
    print("train model")
    start = datetime.now()
    print("Begin time: {}".format(start.isoformat(timespec='seconds')))
    print("Args:{}".format(args))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = Font2FontCGAN(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                          input_width=args.image_size, output_width=args.image_size, L1_penalty=args.L1_penalty,
                          Ltv_penalty=args.Ltv_penalty, generator_dim=args.generator_dim,
                              discriminator_dim=args.discriminator_dim)
        model.register_session(sess)
        model.build_model(is_training=True)

        model.train(lr_g=args.lr_g, lr_d=args.lr_d, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder,
                    sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps)

    end = datetime.now()
    print("Ending time: {}".format(end))
    duration = end - start
    print("Duration hours: {}".format(duration.total_seconds() / 3600.0))


def infer_model():
    print("infer model")


parser = argparse.ArgumentParser(description='Train or infer with cGAN model')
parser.add_argument('--mode', dest='mode', required=True, default='train', help='mode: train or infer')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--generator_dim', dest='generator_dim', type=int, default=64,
                    help="generator dim number")
parser.add_argument('--discriminator_dim', dest='discriminator_dim', type=int, default=64,
                    help="discriminator dim number")
parser.add_argument('--L1_penalty', dest='L1_penalty', type=float, default=100.0, help='weight for L1 loss')
parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=int, default=1, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                    help="freeze encoder weights during training")
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=500,
                    help='number of batches in between two checkpoints')
parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.0002, help='learning rate of G net optimize')
parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.0001, help='learning rate of D net optimize')


parser.add_argument('--source_obj', dest='source_obj', help='source obj dataset')
parser.add_argument('--model_dir', dest='model_dir', help='model directory')
parser.add_argument('--save_dir', dest='save_dir', help='save directory')

args = parser.parse_args()


if __name__ == '__main__':
    if args.mode == 'train':
        train_model()
    elif args.mode == 'infer':
        infer_model()
