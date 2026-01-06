from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
import torch
import numpy as np
#CNN
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

dataSet_root = "../data"

def DefineTrainSetTestSet():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])
    trainSet = CIFAR10(root=dataSet_root, train= True, download=True, transform= transform)
    print(trainSet)
    print("\n"+"-"*100+"\n")
    testSet = CIFAR10(root=dataSet_root, train=False, download=True, transform=transform)
    print(testSet)

    return trainSet, testSet

def DefineLoader():
    trainSet, testSet = DefineTrainSetTestSet()

    train_loader = DataLoader(trainSet, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(testSet, batch_size=128, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes
    

def CalcMeanAndStd():
    trainSet, testSet = DefineTrainSetTestSet()
    imgs = [item[0] for item in trainSet] # item[0] imagens e item[1] classe
    imgs = torch.stack(imgs, dim=0).numpy()

    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print(mean_r,mean_g,mean_b)

    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print(std_r,std_g,std_b)


class CNN(nn.Module):
    def __init__(self):
        super().__init__
        # in_channels da primeira camada: sempre 3 (matriz RGB)
        # in_channels das camadas seguintes: igual ao out_channel da camada anterior
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # in_features da primeira camada FC: conta
        # in_features das demais camadas FC: out_features da camada anterior
        self.fc1 = nn.Linear(in_features=64*3*3, out_features=15)
        self.fc2 = nn.Linear(in_features=15, out_features=25)

    def foward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        return self.fc2(x)