# BlockCNN
BlockCNN: A Deep Network for Artifact Removal and Image Compression

![alt text](https://github.com/DaniMlk/BlockCNN/blob/master/Snapshot.png)

This repository containing the implementation of BlockCNN which published in CVPR Workshop 2018. The implementation is in Pytorch.
Original paper can be found in this link http://openaccess.thecvf.com/content_cvpr_2018_workshops/w50/html/Maleki_BlockCNN_A_Deep_CVPR_2018_paper.html

## Requirments:
- Python 3.6
- Pytorhc 0.3
- Torchvision
- Opencv
- Tensorflow 1.3
- Matplotlib

## Instalation:
```
git clone https://github.com/DaniMlk/BlockCNN.git
cd BlockCNN
# [Option 1] To replicate the conda environment:
conda env create -f environment.yml
source activate pytorch
# [Option 2] Install everything globaly.
```
## Using
We used Pascal_VOC 2012 to train our network. To run the code firstly you should put your dataset in the root path which mentioned in the main.py

```
python main.py
```

## Results
In the below images you can see the output of our network, the input image and the original of it. The result of it shows an amazing fact that our network can enhance the quality of the image significantly.

![alt text](https://github.com/DaniMlk/BlockCNN/blob/master/Result.png)
