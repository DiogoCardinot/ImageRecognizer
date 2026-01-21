import torch
import os
from trainingCNN import CNN
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random

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
    correct = 0
    total = 0

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for images,labels in testLoader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label==prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    acuracia_class = {classname: 0 for classname in classes}

    for classname, correct_count in correct_pred.items():
        acuracia_classes = 100*float(correct_count)/total_pred[classname]
        acuracia_class[classname] = acuracia_classes

    acuracia = (correct/total)*100
    return acuracia, acuracia_class


def main():
    set_seed(seed=42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])
    testSet = CIFAR10(root=dataSet_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(testSet, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    acuracia, acuracia_classes = Test(test_loader)

    acura = 0
    for classname, accuracy in acuracia_classes.items():
        print(f"Classe : {classname} -> Acuracia: {accuracy} %")
        acura += accuracy

    print(100*"*")
    print(f"Acuracia media: {acura/10}")
    print(100*"-")
    print(f"Acuracia Geral: {acuracia} %")
    
if __name__=='__main__':
    main()