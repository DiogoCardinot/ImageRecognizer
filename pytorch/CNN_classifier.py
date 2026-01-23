import torch
import os
from trainingCNN import CNN
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import json

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

dataSet_root = "../data"

model_weigths_path = "./model/pytorch_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

try:
    model = CNN()
    state_dict = torch.load(model_weigths_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # print(f"Modelo carregado com sucesso no device: {device}!")
except Exception as e:
    print(f"Erro ao carregar pesos:\n {e}")

def Test(testLoader):
    model.eval()
    correct = 0
    total = 0

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    batch_data_loading_times = []
    batch_compute_times = []

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        time_start_test = time.time()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        for images,labels in testLoader:
            data_time = time.time() - end
            batch_data_loading_times.append(data_time)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_compute_time = time.time()
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()

            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time = time.time() - start_compute_time

            batch_compute_times.append(compute_time)

            for label, prediction in zip(labels, predicted):
                if label==prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        if device.type== "cuda":
            torch.cuda.synchronize()
            peaky_gpu_memory= torch.cuda.max_memory_allocated(device)/ (1024**2)
        time_total_test = time.time() - time_start_test
        

        end = time.time()
    acuracia_class = {classname: 0 for classname in classes}

    for classname, correct_count in correct_pred.items():
        acuracia_classes = 100*float(correct_count)/total_pred[classname]
        acuracia_class[classname] = acuracia_classes

    acuracia = (correct/total)*100
    return acuracia, acuracia_class, time_total_test, batch_data_loading_times, batch_compute_times, peaky_gpu_memory

test_path = './testMetrics'

if not os.path.exists(test_path):
    os.makedirs(test_path)

def saveMetrics(acuracia, acuracia_class, time_total_test, batch_data_loading_times, batch_compute_times, peaky_gpu_memory, test_loader):

    num_images = len(test_loader.dataset)
    images_per_second = num_images/time_total_test

    batch_data_loading_times = np.array(batch_data_loading_times)
    batch_compute_times = np.array(batch_compute_times)

    mean_batch_data_loading_times = batch_data_loading_times.mean()
    std_batch_data_loading_times = batch_data_loading_times.std()
    total_data_loading_time = batch_data_loading_times.sum()

    mean_batch_compute_times = batch_compute_times.mean()
    std_batch_compute_times = batch_compute_times.std()
    total_compute_time = batch_compute_times.sum()

    data_ratio = total_data_loading_time / time_total_test
    compute_ratio = total_compute_time / time_total_test

    #Total de memória da GPU
    total_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    total_mem_mb = total_mem_bytes / 1024**2

    peaky_gpu_memory_percentage = peaky_gpu_memory/total_mem_mb

    metrics_pytorch = {
        'final_accuracy': acuracia,
        'final_accuracy_per_class': acuracia_class,
        'total_time_test': time_total_test,
        'images_per_second': images_per_second,
        'peaky_gpu_memory': peaky_gpu_memory,
        'peaky_peaky_gpu_memory_percentage': peaky_gpu_memory_percentage,
        'data_loading_ratio': data_ratio,
        'compute_ratio': compute_ratio,
        #BATCH
        'Data loading': 'metricas para carregar os dados a cada batch',
        'total_data_loading_time': total_data_loading_time,
        'mean_batch_data_loading_times': mean_batch_data_loading_times,
        'std_batch_data_loading_times': std_batch_data_loading_times,
        # ----------------------------------------------------------------
        'Compute': 'metricas para processar os dados a cada batch',
        'total_compute_time': total_compute_time,
        'mean_batch_compute_times': mean_batch_compute_times,
        'std_batch_compute_times': std_batch_compute_times,
    }
    
    file_save_teste_metrics = os.path.join(test_path, 'pytorch_metrics.json')

    try:
        with open(file_save_teste_metrics, mode='w') as f:
            json.dump(metrics_pytorch, f, indent=4)
        
        print("Métricas salvas com sucesso!")
    except Exception as e:
        print(f'Erro ao salvar os dados: \n{e}')


def main():
    set_seed(seed=42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])
    testSet = CIFAR10(root=dataSet_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(testSet, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    acuracia, acuracia_class, time_total_test, batch_data_loading_times, batch_compute_times, peaky_gpu_memory = Test(test_loader)

    acura = 0
    for classname, accuracy in acuracia_class.items():
        print(f"Classe : {classname} -> Acuracia: {accuracy} %")
        acura += accuracy

    print(100*"*")
    print(f"Acuracia media: {acura/10}")
    print(100*"-")
    print(f"Acuracia Geral: {acuracia} %")

    try:
        saveMetrics(acuracia, acuracia_class, time_total_test, batch_data_loading_times, batch_compute_times, peaky_gpu_memory, test_loader)
    except Exception as e:
        print(f"Erro ao salvar metricas:\n{e}")
    
if __name__=='__main__':
    main()