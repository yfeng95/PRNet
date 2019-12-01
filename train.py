import numpy as np
import os
import argparse
import tensorflow as tf
import cv2
import random
from predictor import resfcn256
import math


class TrainData(object):

    def __init__(self, train_data_file):
        super(TrainData, self).__init__()
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self.train_data_list)

    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip().split()
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        for item in batch_list:
            img = cv2.imread(item[0])
            label = np.load(item[1])

            im_array = np.array(img, dtype=np.float32)
            # imgs.append(img_array/255.0)
            imgs.append(img_array / 256.0 / 1.1)

            label_array = np.array(img, dtype=np.float32)
            # labels_array_norm = (label_array-label_array.min())/(label_array.max()-label_array.min())
            labels.append(label_array / 256 / 1.1)

        batch.append(imgs)
        batch.append(labels)

        return batch

    def __call__(self, batch_num):
        if (self.index + batch_num) <= self.num_data:
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num

            return batch_data
        # elif self.index < self.num_data:
        #     batch_list = self.train_data_list[self.index:self.num_data]
        #     batch_data = self.getBatch(batch_list)
        #     self.index = 0
        #     return batch_data
        else:
            self.index = 0
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num
            return batch_data


def main(args):

    # Some arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    train_data_file = args.train_data_file
    model_path = args.model_path

    save_dir = args.checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training data
    data = TrainData(train_data_file)

    begin_epoch = 0
    if os.path.exists(model_path + '.data-00000-of-00001'):
        begin_epoch = int(model_path.split('_')[-1]) + 1

    epoch_iters = data.num_data / batch_size
    global_step = tf.Variable(epoch_iters * begin_epoch, trainable=False)
    # Declay learning rate half every 5 epochs
    decay_steps = 5 * epoch_iters
    # learning_rate = learning_rate * 0.5 ^ (global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
                                               decay_steps, 0.5, staircase=True)

    x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    label = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])

    # Train net
    net = resfcn256(256, 256)
    x_op = net(x, is_training=True)

    # Loss
    weights = cv2.imread("Data/uv-data/weight_mask_final.jpg")  # [256, 256, 3]
    weights_data = np.zeros([1, 256, 256, 3], dtype=np.float32)
    weights_data[0, :, :, :] = weights / 16.0
    loss = tf.losses.mean_squared_error(label, x_op, weights_data)

    # This is for batch norm layer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                            beta1=0.9, beta2=0.999, epsilon=1e-08,
                                            use_locking=False).minimize(loss, global_step=global_step)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    if os.path.exists(model_path + '.data-00000-of-00001'):
        tf.train.Saver(net.vars).restore(sess, model_path)

    saver = tf.train.Saver(var_list=tf.global_variables())
    save_path = model_path

    # Begining train
    for epoch in xrange(begin_epoch, epochs):
        for _ in xrange(int(math.ceil(1.0 * data.num_data / batch_size))):
            batch = data(batch_size)
            loss_res, _, global_step_res, learning_rate_res = sess.run(
                [loss, train_step, global_step, learning_rate], feed_dict={x: batch[0], label: batch[1]})

            print('global_step:%d:iters:%d/epoch:%d,learning rate:%f,loss:%f' % (global_step_res, iters, epoch, learning_rate_res, loss_res))

        saver.save(sess=sess, save_path=save_path + '_' + str(epoch))


if __name__ == '__main__':

    par = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    par.add_argument('--train_data_file', default='face3d/examples/trainDataLabel.txt', type=str, help='The training data file')
    par.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate')
    par.add_argument('--epochs', default=100, type=int, help='Total epochs')
    par.add_argument('--batch_size', default=16, type=int, help='Batch sizes')
    par.add_argument('--checkpoint', default='checkpoint/', type=str, help='The path of checkpoint')
    par.add_argument('--model_path', default='checkpoint/256_256_resfcn256_weight', type=str, help='The path of pretrained model')
    par.add_argument('--gpu', default='0', type=str, help='The GPU ID')

    main(par.parse_args())
