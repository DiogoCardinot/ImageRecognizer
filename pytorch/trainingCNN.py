from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os
#CNN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#REPRODUTIBILIDADE
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


dataSet_root = "../data"
num_epochs = 2
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNN(nn.Module):
    # define as camadas
    def __init__(self):
        super().__init__()
        # in_channels da primeira camada: sempre 3 (matriz RGB)
        # in_channels das camadas seguintes: igual ao out_channel da camada anterior
        # (input size - kernel size) / stride + bias -> bias = 1 (camadas convolucionais)
        # input size != in_channels (input size é o valor da largura e altura que sai da camada anterior)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5) # -> (tamanho do input, largura, altura)  largura, altura = (32-5)/1 +1 -> 28 -> (32,28,28)
        # height/2 and width/2
        self.pool = nn.MaxPool2d(2,2) # (32, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # (14-3)/1 + 1 -> 12 -> (64,12,12) -> pool -> (64, 6, 6)
        # # gap: global average pooling (converte altura e largura para 1) -> evita a necessidade de calculos do valor de in_features da primeira fc layer
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # primeira camada FC: in_features = out_channels da ultima camada convolucional * H * W
        # ultima camada FC: out_features = total de classes
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(in_features=84, out_features=len(classes))
    # tem que ser forward o nome (define como os dados vão passar pela rede)
    def forward(self, x):
        #conv+relu+pool (layer 1)
        x=self.pool(F.relu(self.conv1(x)))
        #conv+relu+pool (layer 2)
        x=self.pool(F.relu(self.conv2(x)))
        #achatamento
        x=torch.flatten(x,1)
        #primeira camada totalmente conectada
        x=F.relu(self.fc1(x))
        #segunda camada totalmente conectada
        x=F.relu(self.fc2(x))
        #dropout
        x = self.dropout(x)
        #terceira camada totalmente conectada
        x = self.fc3(x)
        return x


def train(net, train_loader, optimizer, criterion, num_epochs, device):
    net.train()
    for epoch in range(num_epochs):
        print(f'Epoca {epoch + 1} ---------------------')

        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            #non_blocking=True -> faz copia assincrona, GPU trabalha enquanto os dados chegam
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Valor de Loss = {running_loss / len(train_loader)}\n')


model_folder_path = './model/pytorch_model.pth'
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)

model_save_path = './model/pytorch_model.pth'

def main():
    set_seed(seed=42)
    torch.backends.cudnn.benchmark = False # Desempenho
    torch.backends.cudnn.deterministic = True  # Reprodutibilidade
    #define o dispositivo que vai executar, caso encontre gpu vai usar, caso n, vai usar cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])
    trainSet = CIFAR10(root=dataSet_root, train= True, download=True, transform= transform)
    # print(trainSet)
    # print("\n"+"-"*100+"\n")
    
    train_loader = DataLoader(trainSet, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    #envia a arquitetura para o device (gpu ou cpu)
    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(net, train_loader, optimizer, criterion, num_epochs, device)

    try:
        torch.save(net.state_dict(), model_save_path)
        print("Modelo salvo com sucesso!")
    except Exception as e:
        print("Erro ao salvar modelo!")
        print(f'Detalhes: \n {e}')


if __name__ == '__main__':
    main()