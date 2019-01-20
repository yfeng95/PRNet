import numpy as np
import os 
import argparse
import tensorflow as tf
import cv2
import random
from predictor import resfcn256
import math

class TrainData(object):

	def __init__(self,train_data_file):
		super(TrainData,self).__init__()
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

	def getBatch(self,batch_list):
		batch = []
		imgs = []
		labels = []
		for item in batch_list:
			img = cv2.imread(item[0])
			label = np.load(item[1])
			
			im_array = np.array(img,dtype=np.float32)
			imgs.append(img_array/255.0)

			label_array = np.array(img,dtype=np.float32)
			labels_array_norm = (label_array-label_array.min())/(label_array.max()-label_array.min())
			labels.append(labels_array_norm)			

		batch.append(imgs)
		batch.append(labels)

		return batch

	def __call__(self,batch_num):
		if (self.index+batch_num) <= self.num_data:
			batch_list = self.train_data_list[self.index:(self.index+batch_num)]
			batch_data = self.getBatch(batch_list)
			self.index += batch_num

			return batch_data
		elif self.index < self.num_data:
			batch_list = self.train_data_list[self.index:self.num_data]
			batch_data = self.getBatch(batch_list)
			self.index = 0
			return batch_data
		else:
			self.index = 0
			batch_list = self.train_data_list[self.index:(self.index+batch_num)]
			batch_data = self.getBatch(batch_list)
			self.index += batch_num
			return batch_data


def main(args):
	
	# Some arguments
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	batch_size = args.batch_size
	epochs = args.epochs
	train_data_file = args.train_data_file
	learning_rate = args.learning_rate
	model_path = args.model_path

	save_dir = args.checkpoint
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	# Training data
	data = TrainData(train_data_file)
	
	x = tf.placeholder(tf.float32,shape=[None,256,256,3])
	label = tf.placeholder(tf.float32,shape=[None,256,256,3])

	# Train net
	net = resfcn256(256,256)
	x_op = net(x,is_training=True)
	
	# Loss
	loss = tf.losses.mean_squared_error(label,x_op)
	
	# This is for batch norm layer
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
	sess.run(tf.global_variables_initializer())
	
	if os.path.exists(model_path):
		tf.train.Saver(net.vars).restore(sess.model_path)

	saver = tf.train.Saver(var_list = tf.global_variables())
	save_path = model_path
	
	# Begining train
	for epoch in xrange(epochs):
		for _ in xrange(int(math.ceil(1.0*data.num_data/batch_size))):
			batch = data(batch_size)			
			loss_res = sess.run(loss,feed_dict={x:batch[0],label:batch[1]})
			sess.run(train_step,feed_dict={x:batch[0],label:batch[1]})

			print('iters:%d/epoch:%d,learning rate:%f,loss:%f'%(iters,epoch,learn_rate,loss_res))
		
		saver.save(sess=sess,save_path=save_path)


if __name__ == '__main__':

	par = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

	par.add_argument('--train_data_file',default='Data/trainData/trainDataLabel.txt',type=str,help='The training data file')
	par.add_argument('--learning_rate',default=0.0001,type=float,help='The learning rate')
	par.add_argument('--epochs',default=200,type=int,help='Total epochs')
	par.add_argument('--batch_size',default=16,type=int,help='Batch sizes')
	par.add_argument('--checkpoint',default='checkpoint/',type=str,help='The path of checkpoint')
	par.add_argument('--model_path',default='checkpoint/256_256_resfcn256_weight',type=str,help='The path of pretrained model')
	par.add_argument('--gpu',default='0',type=str,help='The GPU ID')

	main(par.parse_args())









































































	





























































































			
		
