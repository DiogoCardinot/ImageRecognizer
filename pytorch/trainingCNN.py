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

#TIME
import time

#DATA SAVE
import json

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

dataSet_root = "../data"
num_epochs = 100
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
    time_per_epoch = []
    peaky_gpu_memory_per_epoch = []
    loss_per_epoch = []
    accuracy_training_per_epoch = []
    #dados por batch
    data_loading_batch_times = {}  #tempo total para converter os dados e fazer o processo
    compute_batch_times = {} #tempo necessário para o processamento, desconsiderando a conversão dos dados
    loss_batch = {}

    for epoch in range(num_epochs):
        print(f'Epoca {epoch + 1} ---------------------')
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        time_start_epoch = time.time()

        running_loss = 0.0
        total = 0
        correct = 0

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        data_loading_batch_times[epoch + 1] = []
        compute_batch_times[epoch+1] = []
        loss_batch[epoch+1] = []

        if epoch == num_epochs-1:
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

        for images, labels in train_loader:
            data_time = time.time()-end
            data_loading_batch_times[epoch + 1].append(data_time)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_compute_time = time.time()
            #non_blocking=True -> faz copia assincrona, GPU trabalha enquanto os dados chegam
            optimizer.zero_grad()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = time.time() - start_compute_time
            compute_batch_times[epoch+1].append(compute_time)

            loss_batch[epoch+1].append(loss.item())
            running_loss += loss.item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            if epoch==num_epochs-1:
                for label, prediction in zip(labels, predicted):
                    if label==prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
                
        if device.type == "cuda":
            torch.cuda.synchronize()
            peaky_gpu_memory_per_epoch.append(torch.cuda.max_memory_allocated(device)/ (1024**2))

        if epoch == num_epochs-1:
            accuracy_per_class = {classname: 0 for classname in classes}
            for classname, correct_count in correct_pred.items():
                accuracy_class = 100*(correct_count)/total_pred[classname]
                accuracy_per_class[classname] = accuracy_class

        time_total_epoch = time.time() - time_start_epoch
        time_per_epoch.append(time_total_epoch)
        loss_per_epoch.append(running_loss / len(train_loader))
        accuracy_training_per_epoch.append(correct/total)

        print(f'Valor de Loss = {running_loss / len(train_loader)}\n')

    return time_per_epoch, data_loading_batch_times, compute_batch_times, peaky_gpu_memory_per_epoch, loss_per_epoch, accuracy_training_per_epoch, loss_batch, accuracy_per_class


model_folder_path = './model/pytorch_model.pth'
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)

model_save_path = './model/pytorch_model.pth'

save_path = "./trainingMetrics"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def saveMetrics(time_per_epoch, data_loading_batch_times, compute_batch_times, peaky_gpu_memory_per_epoch, loss_per_epoch, accuracy_training_per_epoch, loss_batch, accuracy_per_class, train_loader, device):
    time_per_epoch = np.array(time_per_epoch)
    peaky_gpu_memory_per_epoch= np.array(peaky_gpu_memory_per_epoch)
    loss_per_epoch = np.array(loss_per_epoch)
    accuracy_training_per_epoch = np.array(accuracy_training_per_epoch)
    
    mean_time_per_epoch = time_per_epoch.mean()
    std_time_per_epoch = time_per_epoch.std()
    total_time_training = time_per_epoch.sum()
    num_images = len(train_loader.dataset) * num_epochs
    images_per_second = num_images / total_time_training
    images_per_second_epoch = [len(train_loader.dataset)/t for t in time_per_epoch]
    images_per_second_epoch = np.array(images_per_second_epoch)
    time_per_batch = total_time_training / (len(train_loader)*num_epochs)

    #Total de memória da GPU
    total_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    total_mem_mb = total_mem_bytes / 1024**2

    percent_usage_memory = [(m/total_mem_mb) for m in peaky_gpu_memory_per_epoch]
    mean_percentage_usage_memory = np.mean(percent_usage_memory)
    std_percentage_usage_memory = np.std(percent_usage_memory)

    #Batch Data
    mean_batch_data_loading_time_per_epoch = []
    std_batch_data_loading_time_per_epoch = []
    total_batch_data_loading_time_per_epoch = []
    
    mean_batch_compute_time_per_epoch = []
    std_batch_compute_time_per_epoch = []
    total_time_batch_compute_per_epoch = []
    
    for epoch in range(num_epochs):
        epoch_data_loading_batch_times = np.array(data_loading_batch_times[epoch+1])
        mean_batch_data_loading_time_per_epoch.append(epoch_data_loading_batch_times.mean())
        std_batch_data_loading_time_per_epoch.append(epoch_data_loading_batch_times.std())
        total_batch_data_loading_time_per_epoch.append(epoch_data_loading_batch_times.sum())

        epoch_compute_batch_times = np.array(compute_batch_times[epoch+1])
        mean_batch_compute_time_per_epoch.append(epoch_compute_batch_times.mean())
        std_batch_compute_time_per_epoch.append(epoch_compute_batch_times.std())
        total_time_batch_compute_per_epoch.append(epoch_compute_batch_times.sum())

    total_time_data_loading = np.array(total_batch_data_loading_time_per_epoch).sum()
    total_time_compute = np.array(total_time_batch_compute_per_epoch).sum()

    data_ratio = total_time_data_loading / total_time_training
    compute_ratio = total_time_compute / total_time_training

    metrics_pytorch = {
        #Time per epoch
        'TEMPO POR EPOCA': 'usado para identificar as informacoes por epoca',
        'time_per_epoch': time_per_epoch.tolist(),
        'mean_time_per_epoch': mean_time_per_epoch,
        'std_time_per_epoch': std_time_per_epoch,
        'total_time_training': total_time_training,
        'total_time_data_loading': total_time_data_loading,
        'total_time_compute': total_time_compute,
        'images_per_second': images_per_second,
        'images_per_second_epoch': images_per_second_epoch.tolist(),
        'time_per_batch': time_per_batch,
        'data_loading_ratio': data_ratio,
        'compute_ratio': compute_ratio,
        #GPU peaky memory
        'Maximo uso da memeria da GPU':'mensura o maior valor de uso da memória da GPU por epoca',
        'peaky_gpu_memory_per_epoch': peaky_gpu_memory_per_epoch.tolist(),
        'mean_peaky_gpu_memory': peaky_gpu_memory_per_epoch.mean(),
        'std_peaky_gpu_memory': peaky_gpu_memory_per_epoch.std(),
        'mean_usage_gpu_memory_percentage': mean_percentage_usage_memory,
        'std_usage_gpu_memory': std_percentage_usage_memory,
        #LOSS   
        'Valores de loss': 'verificar convergencia do modelo',
        'mean_loss': loss_per_epoch.mean(),
        'std_loss': loss_per_epoch.std(),
        'loss_per_epoch': loss_per_epoch.tolist(),
        #Accuracy Training
        'Acuracia do treinamento': 'verificar a porcentagem de acerto do treinamento',
        'accuracy_training_per_epoch_vector': accuracy_training_per_epoch.tolist(), #usar para identificar a quantidade de épocas necessárias até chegar em um determinado valor
        'mean_accuracy_training_per_epoch_percentage': accuracy_training_per_epoch.mean(),
        'std_accuracy_training_per_epoch': accuracy_training_per_epoch.std(),
        'accuracy_per_class': {str(k): v for k,v in accuracy_per_class.items()},
        # Batch data time
        'TEMPO DE MANIPULACAO/CARREGAMENTO DOS DADOS POR BATCH':'tempo necessario para enviar os dados para o device',
        'mean_batch_data_loading_time_per_epoch': mean_batch_data_loading_time_per_epoch,
        'std_batch_data_loading_time_per_epoch': std_batch_data_loading_time_per_epoch,
        'total_batch_data_loading_time_per_epoch': total_batch_data_loading_time_per_epoch,
        'data_loading_batch_times': {str(k): v for k, v in data_loading_batch_times.items()},
        #Batch compute time
        'TEMPO DE TREINAMENTO DE FATO':'tempo necessario para realizar o treinamento',
        'mean_batch_compute_time_per_epoch': mean_batch_compute_time_per_epoch,
        'std_batch_compute_time_per_epoch': std_batch_compute_time_per_epoch,
        'total_batch_compute_per_epoch': total_time_batch_compute_per_epoch,
        'compute_batch_times': {str(k): v for k, v in compute_batch_times.items()},
        #Batch loss
        'Batch loss': 'Loss de cada batch de cada epoca',
        'loss_batch': {str(k): v for k, v in loss_batch.items()},
    }
    file_save_metrics_path = os.path.join(save_path, 'pytorch_metrics.json')

    try:
        with open(file_save_metrics_path, mode='w') as f:
            json.dump(metrics_pytorch, f, indent=4)
        
        print("Métricas salvas com sucesso!")
    except Exception as e:
        print(f'Erro ao salvar os dados: \n{e}')
    

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

    time_per_epoch, data_loading_batch_times, compute_batch_times, peaky_gpu_memory_per_epoch, loss_per_epoch, accuracy_training_per_epoch, loss_batch, accuracy_per_class = train(net, train_loader, optimizer, criterion, num_epochs, device)

    try:
        torch.save(net.state_dict(), model_save_path)
        print("Modelo (CNN) salvo com sucesso!")
    except Exception as e:
        print("Erro ao salvar modelo (CNN)!")
        print(f'Detalhes: \n {e}')
    
    try:
        saveMetrics(time_per_epoch, data_loading_batch_times, compute_batch_times, peaky_gpu_memory_per_epoch, loss_per_epoch, accuracy_training_per_epoch, loss_batch, accuracy_per_class, train_loader, device)
    except Exception as e:
        print(f'Erro ao salvar as métricas:\n{e}')


if __name__ == '__main__':
    main()