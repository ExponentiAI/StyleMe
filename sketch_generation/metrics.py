# example of calculating the frechet inception distance in Keras
import numpy
import os
import cv2
import argparse
import torch
import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input


# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'   # 只显示 Error

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(np.dot(sigma1, sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

#act1 =generatedImg  ,act2 = realImg
def calculate_fid_modify(act1,act2):
	# calculate activations
	# act1 = model.predict(images1)
	# act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(np.dot(sigma1, sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def data_list(dirPath):
	generated_Dataset = []
	real_Dataset = []
	for root, dirs, files in os.walk(dirPath):
		for filename in sorted(files):  # sorted已排序的列表副本
			# 判断该文件是否是目标文件
			if "generated" in filename:
				generatedPath = root + '/' + filename
				generatedImg = cv2.imread(generatedPath).astype('float32')
				generated_Dataset.append(generatedImg)
				# 对比图片路径
				realPath = root + '/' + filename.replace('generated', 'real')
				realImg = cv2.imread(realPath).astype('float32')
				real_Dataset.append(realImg)
	return generated_Dataset, real_Dataset

if __name__ == '__main__':
	### 参数设定
	parser = argparse.ArgumentParser()
	# parser.add_argument('--dataset_dir', type=str, default='./results/hrnet/', help='results')
	parser.add_argument('--dataset_dir', type=str, default='./results/ssngan/', help='results')
	parser.add_argument('--name', type=str, default='sketch', help='name of dataset')
	opt = parser.parse_args()

	# 数据集
	dirPath = os.path.join(opt.dataset_dir, opt.name)
	generatedImg, realImg = data_list(dirPath)
	dataset_size = len(generatedImg)
	print("数据集：", dataset_size)

	images1 = torch.Tensor(generatedImg)
	images2 = torch.Tensor(realImg)
	print('shape: ', images1.shape, images2.shape)

	# 将全部数据集导入
	# prepare the inception v3 model
	model = InceptionV3(include_top=False, pooling='avg')

	# pre-process images(归一化)
	images1 = preprocess_input(images1)
	images2 = preprocess_input(images2)

	# fid between images1 and images2
	fid = calculate_fid(model, images1, images2)
	print('FID : %.3f' % fid)
	print('FID_average : %.3f' % (fid / dataset_size))



