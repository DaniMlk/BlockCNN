import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        k = 64

        self.conv_1 = nn.Conv2d(color, k, (3, 5), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(k)

        self.layer_1 = BottleNeck(k, k)
        self.layer_2 = BottleNeck(k, k)

        self.conv_2 = nn.Conv2d(k, k*2, (3, 5), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(k*2)

        self.layer_3 = BottleNeck(k*2, k*2)

        self.conv_3 = nn.Conv2d(k*2, k*4, (1, 5), (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(k*4)

        self.layer_4 = BottleNeck(k*4, k*4)
        self.layer_5 = BottleNeck(k*4, k*4)

        self.conv_4 = nn.Conv2d(k*4, k*8, (1, 1), (1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(k*8)
        
        self.layer_6 = BottleNeck(k*8, k*8)

        self.conv_5 = nn.Conv2d(k*8, k*4, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(k*4)

        self.layer_7 = BottleNeck(k*4, k*4)

        self.conv_6 = nn.Conv2d(k*4, k*2, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(k*2)

        self.layer_8 = BottleNeck(k*2, k*2)

        self.conv_7 = nn.Conv2d(k*2, k, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(k)

        self.layer_9 = BottleNeck(k, k)

        self.conv_8 = nn.Conv2d(k, color, 1, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.relu(self.bn2(self.conv_2(x)))
        x = self.layer_3(x)
        x = self.relu(self.bn3(self.conv_3(x)))
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.relu(self.bn4(self.conv_4(x)))
        x = self.layer_6(x)
        x = self.relu(self.bn5(self.conv_5(x)))
        x = self.layer_7(x)
        x = self.relu(self.bn6(self.conv_6(x)))
        x = self.layer_8(x)
        x = self.relu(self.bn7(self.conv_7(x)))
        x = self.layer_9(x)
        x = self.conv_8(x)
        x = self.sig(x)
        x = x * 255.0

        return x
