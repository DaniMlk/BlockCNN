import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from logger import Logger
import os
import argparse
import shutil
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
from torchvision import transforms
from model import CNN_Net

# Constants
PICTURE_SIZE = 320
BATCH_SIZE = 32
COLOR = 3
IN_SIZE = 8
OUT_SIZE = 24
ITERATE_SIZE = int((PICTURE_SIZE - (2 * IN_SIZE)) / IN_SIZE) 
NUM_PER_PIC = int(((PICTURE_SIZE - 2 * IN_SIZE) ** 2) / (IN_SIZE ** 2))
TOTAL_PIC = 3000
TOTAL_PIC_TEST = 10
FILTER_K = 64
NUM_PIC_SHOW = 1
BEST_LOSS = 1e8
ROOT = "/home/mlcm/Danial/Image_compression/dataset"

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Image Generating')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('-o', '--save_dir', type=str, default='./save', help='Location for parameter checkpoints and samples')
    return parser.parse_args()

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_np(x):
    return x.data.cpu().numpy()

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

class My_dataloader(Dataset):
    def __init__(self, data_24, transform):
        self.data_24 = data_24
        self.pathes_24 = list(glob(self.data_24))
        self.transform = transform

    def __len__(self):
        return len(self.pathes_24)

    def __getitem__(self, idx):
        img_24 = Image.open(self.pathes_24[idx]).convert('RGB')
        if self.transform:
            img_24 = self.transform(img_24)
            img_8 = img_24[:,6:18,6:18]
        return img_24*255., img_8*255.

def main():
    # Parse arguments
    args = parse_args()

    # Prepare directories
    create_dir(args.save_dir)
    create_dir(os.path.join(args.save_dir, "Generated_Pic/train"))
    create_dir(os.path.join(args.save_dir, "Generated_Pic/test"))
    create_dir(os.path.join(args.save_dir, "checkpoint"))
    create_dir(os.path.join(args.save_dir, "logs"))

    # Load model
    model = CNN_Net(FILTER_K).cuda()
    model.apply(weight_init)

    # Load loss function
    loss_fn = nn.MSELoss().cuda()

    # Load optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Set up data loader
    transform = transforms.Compose([
        transforms.Resize((PICTURE_SIZE, PICTURE_SIZE)),
        transforms.ToTensor()
    ])

    # Set up data set
    train_data_24 = My_dataloader(ROOT + "/train/*.png", transform)
    train_loader = DataLoader(train_data_24, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # Train model
    for epoch in range(500):
        for i, (data_24, data_8) in enumerate(train_loader):
            optimizer.zero_grad()
            input = data_24[:,:,0:16,:].clone().detach().cuda().float()
            input[:,:,8:16,8:24] = input[:,:,0:8,0:24].mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).expand_as(input[:,:,8:16,8:24])
            target = data_8.clone().detach().cuda().float()
            out = model(input)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

    # Save checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(args.save_dir, 'checkpoint', 'checkpoint.pth.tar'))

if __name__ == "__main__":
    main()
