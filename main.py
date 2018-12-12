import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb
import cv2
from logger import Logger
import os.path
import argparse
import shutil
from matplotlib import style
import itertools
import random
from torch.utils.data import Dataset, DataLoader
from glob import glob
from skimage import io, transform
from torchvision import transforms
from model import CNN_Net
style.use('ggplot')
global args
parser = argparse.ArgumentParser(description='PyTorch Image Generating')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('-o', '--save_dir', type=str, default='./save', help='Location for parameter checkpoints and samples')

############################### Hyper_Parameter_of_Our_Net ####################################
picture_size = 320
batch_size = 32
color = 3
in_size = 8
out_size = 24
iterate_size = int((picture_size - (2 * in_size))/ in_size) 
num_per_pic = int(((picture_size - 2 * in_size) ** 2) / (in_size**2))
total_pic = 3000
total_pic_test = 10
loss_train =[]
loss_test_mean_total =[]
filter_k = 64
num_pic_show = 1
is_best = True 
best_loss = 100000000
root = "/home/mlcm/Danial/Image_compression/dataset"
args = parser.parse_args()

def to_np(x):
	return x.data.cpu().numpy()


def weight_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def save_checkpoint(state, is_best, filename=args.save_dir+'/checkpoint/checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, args.save_dir+'/checkpoint/model_best.pth.tar')

def freeze_bn(m):
	if isinstance(m, nn.BatchNorm2d):
		m.eval()

def release_weight(model):
	for param in model.parameters():
		param.requires_grad = True
	return model

def freeze_bn_all(model):
	model.apply(freeze_bn)
	return model

def freeze_weight(model):
	for param in model.parameters():
		param.requires_grad = False
	return model

################ Define_Our_customized_DataLoader #########################
class My_dataloader(Dataset):

	def __init__(self, data_24, transform):
		"""
		Args:
			data_24: path to input data
			data_8: path to output data
		"""
		self.data_24 = data_24
		# self.data_8 = data_8
		# print(self.data_24)
		self.pathes_24 = list(glob(self.data_24))
		# print(self.pathes_24)
		# self.pathes_8 = list(glob(self.data_8))
		self.transform = transform

	def __len__(self):
		return int((len(self.pathes_24)*8)/8)

	def __getitem__(self, idx):
		img_24 = Image.open(self.pathes_24[idx]).convert('RGB')
		# img_8 = Image.open(self.pathes_8[idx]).convert('RGB')

		if self.transform:
			img_24 = self.transform(img_24)
			img_8 = img_24[:,6:18,6:18]

		return img_24*255., img_8*255.


trans = transforms.Compose([transforms.ToTensor()])
data_loader = My_dataloader(root+'/24_24_high/*.jpg',trans)
train_loader = torch.utils.data.DataLoader(data_loader,batch_size=batch_size, shuffle=True, num_workers = 6, pin_memory=True)
count = 0
model = CNN_Net().cuda()
model.apply(weight_init)
# model = nn.parallel.DataParallel(model,device_ids=[0,1]) #for multi gpu
torch.manual_seed(1)
# optimizer = optim.SGD(model.parameters(),weight_decay=0.0001, lr=0.001,momentum=0.9)
optimizer = optim.Adam(model.parameters(),weight_decay=0.0001, lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
epochSize = 500
loss_t_train = 0

if not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)

if not os.path.exists(args.save_dir+"/Generated_Pic/train"):
	os.makedirs(args.save_dir+"/Generated_Pic/train")

if not os.path.exists(args.save_dir+"/Generated_Pic/test"):
	os.makedirs(args.save_dir+"/Generated_Pic/test")

if not os.path.exists(args.save_dir+"/checkpoint"):
	os.makedirs(args.save_dir+"/checkpoint")

#########################making folder for generated cache#######################
if args.resume:
	if os.path.isfile(args.resume):
		print("=> Loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> Loaded checkpoint '{}' (epoch {})"
			  .format(args.resume, checkpoint['epoch']))
	else:
		print("=> No checkpoint found at '{}'".format(args.resume))

logger = Logger(args.save_dir+'/logs')
print('Start Training ....')
for e in range(epochSize):
	for i, (data_24,data_8) in enumerate(train_loader):
		optimizer.zero_grad()
		input = torch.autograd.Variable(data_24[:,:,0:16,:]).type('torch.FloatTensor').cuda()
		input[:,:,8:16,8:24] = input[:,:,0:8,0:24].mean(dim=-1,keepdim=True).mean(dim=-2,keepdim=True).expand_as(input[:,:,8:16,8:24])
		target = torch.autograd.Variable(data_8).type('torch.FloatTensor').cuda()
		out = model(input)
		loss = nn.SmoothL1Loss()
		loss = loss(out,target)
		loss_t_train += loss.data[0]
		if i%100 ==0:
			print('Loss data for batch '+str(i).rjust(4,'0')+' is = ',loss.data[0]*255.)

		loss.backward()
		optimizer.step()

	if e%1==0:
		loss_t_train = loss_t_train / (total_pic*(num_per_pic/batch_size)) 
		print("Loss in Train => epoch:{},   loss:{}".format(e, loss_t_train*255.))

		################################ Logging ####################################
		info = {
			'loss_train':loss_t_train,
		}

		for tag, value in info.items():
			logger.scalar_summary(tag, value, e)
		for tag, value in model.named_parameters():
			tag = tag.replace('.', '/')
			logger.histo_summary(tag, to_np(value), e)
			logger.histo_summary(tag+'/grad', to_np(value.grad), e)
			############################ End_of_Logging ###############################

		loss_train.append(loss_t_train)
		if e !=0:
			t = np.arange(0,e+1,1)
			plt.plot(t, loss_train)

			plt.xlabel('epoch')
			plt.ylabel('Loss')
			plt.title('Train and test loss for JPEG quality enhancement')
			plt.grid(True)
			# plt.savefig("train_Loss.png")
		loss_t_train = 0
		loss_test_mean = 0
		############################################# Test_Phase ########################
		model.train(False)
		for b in range(total_pic_test):
			data = cv2.imread('/home/mlcm/Danial/Image_compression/dataset/kodak/'+str(b)+'.png')
			# pdb.set_trace()
			input = torch.autograd.Variable(torch.from_numpy(np.zeros((1,3,data.shape[0],data.shape[1]))),volatile=True).type('torch.FloatTensor').cuda()
			output = torch.autograd.Variable(torch.ones((1,3,data.shape[0],data.shape[1])),volatile=True).type('torch.FloatTensor').cuda()
			# data = data/255
			# data = cv2.cvtColor(data,cv2.COLOR_BGR2Lab)
			data = torch.autograd.Variable(torch.from_numpy(data),volatile=True).unsqueeze(0).permute(0,3,1,2).type('torch.FloatTensor').cuda()
			loss_test_t_smooth = 0
			loss_test_t_MSE = 0
			for j in range(int((data.shape[2]-16)/8)):
				for i in range(int((data.shape[3]-16)/8)):
					input[:,:,j*8+8:j*8+16,i*8+8:i*8+24] = input[:,:,j*8:j*8+8,i*8:i*8+24].mean(dim=-1,keepdim=True).mean(dim=-2,keepdim=True).expand_as(input[:,:,j*8+8:j*8+16,i*8+8:i*8+24])
					out = model(input[:,:,j*8:j*8+16,i*8:i*8+24])

					loss = nn.SmoothL1Loss()
					loss_test = loss(out,data[:,:,(j+1)*8-2:(j+1)*8+10,(i+1)*8-2:(i+1)*8+10])
					loss_test_t_smooth += loss_test.data[0]
					loss = nn.MSELoss()
					loss_test = loss(out,data[:,:,(j+1)*8-2:(j+1)*8+10,(i+1)*8-2:(i+1)*8+10])
					loss_test_t_MSE += loss_test.data[0]
					output[:,:,(j+1)*8:(j+1)*8+8,(i+1)*8:(i+1)*8+8] = out[:,:,2:10,2:10]
					input[:,:,(j+1)*8:(j+1)*8+8,(i+1)*8:(i+1)*8+8] =  data[:,:,(j+1)*8:(j+1)*8+8,(i+1)*8:(i+1)*8+8]
			loss_test_mean += loss_test_t_smooth
			print("Test Loss for Image{}/{} ---- MSE:{}    SmoothL1:{}".format(b+1,total_pic_test,(loss_test_t_MSE*255./(38*38)),(loss_test_t_smooth*255./(38*38))))
			output = output.cpu()
			output_plot = output.data.permute(0,2,3,1).type(torch.FloatTensor).numpy()
			cv2.imwrite(args.save_dir+'/Generated_Pic/test/pic'+str(b).rjust(3,'0')+'_epoch_'+str(e)+'.png',output_plot[0])
			print('image test_'+str(b)+' done')

		loss_test_mean = loss_test_mean/(total_pic_test*38*38)
		print("Avg loss in test ",loss_test_mean)
		info = {
			'loss_test_mean':loss_test_mean,
		}

		for tag, value in info.items():
			logger.scalar_summary(tag, value, e)
		loss_test_mean_total.append(loss_test_mean)
		if e !=0:
			t = np.arange(0,e+1,1)
			plt.plot(t, loss_test_mean_total)

			plt.xlabel('epoch')
			plt.ylabel('Loss')
			plt.title('Train and test loss for JPEG quality enhancement')
			plt.grid(True)
			plt.savefig(args.save_dir+"/Loss.png")
		is_best = loss_test_mean < best_loss
		best_loss = min(best_loss, loss_test_mean)

		model.train(True)
		
		save_checkpoint({
			'epoch': e + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			},is_best)

