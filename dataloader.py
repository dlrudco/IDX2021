import torch
from .dataset import RespiDataset, RespiWAVDataset
from torch.utils.data import DataLoader
import random
from random import randint, shuffle
import math
import pickle
import os

class NFold_RespiDataset:
	def __init__(self, path, seed, fold_N, num_mel=128, multi_label=False, fixed_length=None, return_wav=False):
		if seed == -1:
			self.seed = randint(100000, 999999)
		else:
			self.seed = seed
		self.fold_N = fold_N
		self.path = path
		self.num_mel = num_mel
		self.multi_label = multi_label
		self.fixed_length = fixed_length
		self.return_wav = return_wav
		self.total_data = RespiDataset(split='train', data_dir=self.path, initialize=True, 
			num_mel=self.num_mel, multi_label=self.multi_label, 
			mean=None, std=None, fixed_length=self.fixed_length)

		data_num = self.total_data.__len__()
		index_list = list(range(data_num))
		random.seed(self.seed)
		random.shuffle(index_list)
		self.index_chunks = []
		self.sizes = []
		self.chunk_size = math.ceil(data_num / fold_N)
		for n in range(fold_N):
			try:
				self.index_chunks.append(index_list[(n)*self.chunk_size:(n+1)*self.chunk_size])
			except IndexError:
				self.index_chunks.append(index_list[(n)*self.chunk_size:])
			self.sizes.append(len(self.index_chunks[n]))
		print(self.sizes)
		os.makedirs('foldlists', exist_ok=True)
		pickle.dump(self.index_chunks, open(f'foldlists/{self.seed}.pkl','wb'))

	def get_fold(self, k):
		if k >= self.fold_N:
			raise IndexError(f"k should be lower than {self.fold_N}")
		else:
			val_list = []
			train_list = []
			for n in range(self.fold_N):
				if n == k:
					val_list.extend(self.index_chunks[n])
				else:
					train_list.extend(self.index_chunks[n])
			train_data = RespiDataset(split='train', data_dir=self.path, initialize=False, 
			num_mel=self.num_mel, multi_label=self.multi_label, 
			mean=self.total_data.mean, std=self.total_data.std, fixed_length=self.fixed_length)

			train_data.train_data = [self.total_data.data[j] for j in train_list]
			train_data.labels = [self.total_data.labels[j] for j in train_list]
			train_data.data = train_data.train_data

			val_data = RespiDataset(split='val', data_dir=self.path, initialize=False, 
			num_mel=self.num_mel, multi_label=self.multi_label, 
			mean=self.total_data.mean, std=self.total_data.std, fixed_length=self.fixed_length)
			val_data.val_data = [self.total_data.data[k] for k in val_list]
			val_data.labels = [self.total_data.labels[k] for k in val_list]
			val_data.data = val_data.val_data
			return train_data, val_data

class NFold_RespiWAVDataset:
	def __init__(self, path, seed, fold_N, num_mel=128, multi_label=False, fixed_length=None, return_wav=False):
		if seed == -1:
			self.seed = randint(100000, 999999)
		else:
			self.seed = seed
		self.fold_N = fold_N
		self.path = path
		self.num_mel = num_mel
		self.multi_label = multi_label
		self.fixed_length = fixed_length
		self.return_wav = return_wav
		self.total_data = RespiWAVDataset(split='train', data_dir=self.path, initialize=True, 
			num_mel=self.num_mel, multi_label=self.multi_label, 
			mean=None, std=None, fixed_length=self.fixed_length)

		data_num = self.total_data.__len__()
		index_list = list(range(data_num))
		random.seed(self.seed)
		random.shuffle(index_list)
		self.index_chunks = []
		self.sizes = []
		self.chunk_size = math.ceil(data_num / fold_N)
		for n in range(fold_N):
			try:
				self.index_chunks.append(index_list[(n)*self.chunk_size:(n+1)*self.chunk_size])
			except IndexError:
				self.index_chunks.append(index_list[(n)*self.chunk_size:])
			self.sizes.append(len(self.index_chunks[n]))
		print(self.sizes)
		os.makedirs('foldlists', exist_ok=True)
		pickle.dump(self.index_chunks, open(f'foldlists/{self.seed}.pkl','wb'))

	def get_fold(self, k):
		if k >= self.fold_N:
			raise IndexError(f"k should be lower than {self.fold_N}")
		else:
			val_list = []
			train_list = []
			for n in range(self.fold_N):
				if n == k:
					val_list.extend(self.index_chunks[n])
				else:
					train_list.extend(self.index_chunks[n])
			train_data = RespiWAVDataset(split='train', data_dir=self.path, initialize=False, 
			num_mel=self.num_mel, multi_label=self.multi_label, fixed_length=self.fixed_length)

			train_data.train_data = [self.total_data.data[j] for j in train_list]
			train_data.labels = [self.total_data.labels[j] for j in train_list]
			train_data.data = train_data.train_data

			val_data = RespiWAVDataset(split='val', data_dir=self.path, initialize=False, 
			num_mel=self.num_mel, multi_label=self.multi_label, fixed_length=self.fixed_length)
			val_data.val_data = [self.total_data.data[k] for k in val_list]
			val_data.labels = [self.total_data.labels[k] for k in val_list]
			val_data.data = val_data.val_data
			return train_data, val_data
