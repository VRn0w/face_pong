from glob import glob
from random import shuffle

import cv2
import numpy as np 

class DataReader(object):
	def __init__(self, image_directory = '/data/datasets/face/expression/kaggle/fer2013/images/'):
		self.image_directory = image_directory

		self.filenames = None

		self._parseImages()

		print 'found',len(self.filenames),'images'

	def _parseImages(self):
		if self.filenames is None:
			self.filenames = glob(self.image_directory + '*')
			# shuffling dataset
			shuffle(self.filenames)

		emotion_dict = { 'Angry':0,'Disgust':1,'Fear':2,'Happy':3,'Sad':4,'Surprise':5,'Neutral':6}

		self.images = {'train':[],'test':[]}
		self.labels = {'train':[],'test':[]}
		labels_idx = {'train':[],'test':[]}
		
		# read images
		for fn in self.filenames:
			try:
				# cut out directory
				id = fn.split('/')[-1]
				# cut out file extension
				id = '.'.join(id.split('.')[:-1])
				# format is DATASETSPLIT_COUNT_EMOTION
				dataset_split , count, emotion = id.split('_')
				label_idx = emotion_dict[emotion]
				
				# read image
				image = cv2.imread(fn)
				image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

				if np.random.random() > .5:
					image = np.fliplr(image)

				if dataset_split == 'Training':
					self.images['train'].append(image)
					labels_idx['train'].append(label_idx)
				else:
					self.images['test'].append(image)
					labels_idx['test'].append(label_idx)

			except:
				print 'error on',fn
		
		# one hot encoding
		for key in self.images.keys():
			labels_idx[key] = np.array(labels_idx[key],dtype = np.int32)
			self.labels[key] = np.zeros((labels_idx[key].size, labels_idx[key].max()+1))
			self.labels[key][np.arange(labels_idx[key].size),labels_idx[key]] = 1

		self.act_idx = {'train':0,'test':0}

	def getNextBatch(self,batch_size = 64, train = True):
		batch = []
		labels = []

		if train: key_dataset = 'train'
		else: key_dataset = 'test'
		
		while len(batch) < batch_size:
			batch.append(self.images[key_dataset][self.act_idx[key_dataset]])
			labels.append(self.labels[key_dataset][self.act_idx[key_dataset]])
			self.act_idx[key_dataset] = (self.act_idx[key_dataset] + 1) % len(self.images[key_dataset])

		batch = np.array(batch) / 128. - 1.
		if len(batch.shape) == 3:
			batch = np.reshape(batch,(batch.shape[0],batch.shape[1],batch.shape[2],1))
		labels = np.array(labels,dtype = np.int32)

		return batch, labels

if __name__ == '__main__':
	db = DataReader()

	batch, labels = db.getNextBatch()
	print batch.min(),batch.max(),labels.shape # 0.0 1.0 (64, 7)
