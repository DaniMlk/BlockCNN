import torch

color = 3

class BottleNeck(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, inplanes, planes, stride=1):
		super(BottleNeck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(inplanes)
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = F.leaky_relu(out, 0.1)

		out = self.conv2(out)
		out = self.bn2(out)
		out = F.leaky_relu(out, 0.1)

		out = self.conv3(out)
		out = self.bn3(out)

		out += residual
		out = F.leaky_relu(out, 0.1)

		return out
		

def CONV1_1(inplanes,planes,s = 1, p = 0):
		return nn.Conv2d(inplanes, planes, 1, stride = s, padding = p, bias=False)


class CNN_Net(nn.Module):
	"""docstring for CNN_Net"""
	def __init__(self):
		super(CNN_Net, self).__init__()
		k = 64

		self.conv_1 = nn.Conv2d(color, k, (3,5), (1,1), bias=False)
		self.BN1 = nn.BatchNorm2d(k)

		self.layer_1 = BottleNeck(k, k)
		self.layer_2 = BottleNeck(k, k)

		self.conv_2 = nn.Conv2d(k, k*2, (3,5), (1,1), bias=False)
		self.BN2 = nn.BatchNorm2d(k*2)

		self.layer_3 = BottleNeck(k*2, k*2)

		self.conv_3 = nn.Conv2d(k*2, k*4, (1,5), (1,1), bias=False)
		self.BN3 = nn.BatchNorm2d(k*4)

		self.layer_4 = BottleNeck(k*4,k*4)
		self.layer_5 = BottleNeck(k*4,k*4)

		self.conv_4 = nn.Conv2d(k*4, k*8, (1,1), (1,1), bias=False)
		self.BN4 = nn.BatchNorm2d(k*8)
		
		self.layer_6 = BottleNeck(k*8,k*8)

		self.conv_5 = CONV1_1(k*8,k*4)
		self.BN5 = nn.BatchNorm2d(k*4)

		self.layer_7 = BottleNeck(k*4,k*4)

		self.conv_6 = CONV1_1(k*4,k*2)
		self.BN6 = nn.BatchNorm2d(k*2)

		self.layer_8 = BottleNeck(k*2,k*2)

		self.conv_7 = CONV1_1(k*2,k)
		self.BN7 = nn.BatchNorm2d(k)

		self.layer_9 = BottleNeck(k,k)

		self.conv_8 = CONV1_1(k,color)
		self.Sig = nn.Sigmoid()

	def forward(self, x):
		x = F.leaky_relu(self.BN1(self.conv_1(x)), 0.1)
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = F.leaky_relu(self.BN2(self.conv_2(x)), 0.1)
		x = self.layer_3(x)
		x = F.leaky_relu(self.BN3(self.conv_3(x)), 0.1)
		x = self.layer_4(x)
		x = self.layer_5(x)
		x = F.leaky_relu(self.BN4(self.conv_4(x)), 0.1)
		x = self.layer_6(x)
		x = F.leaky_relu(self.BN5(self.conv_5(x)), 0.1)
		x = self.layer_7(x)
		x = F.leaky_relu(self.BN6(self.conv_6(x)), 0.1)
		x = self.layer_8(x)
		x = F.leaky_relu(self.BN7(self.conv_7(x)), 0.1)
		x = self.layer_9(x)
		x = self.conv_8(x)
		x = self.Sig(x)
		x = x * 255.0

		return x