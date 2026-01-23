import os 
import json
import matplotlib.pyplot as plt

training_pytorch_data = "./pytorch/trainingMetrics/pytorch_metrics.json"

with open(training_pytorch_data, mode='r') as file:
    pytorch_data = json.load(file)

# print(pytorch_data['mean_time_per_epoch'])



def ComparativeTable():
    convergencia = 85   #taxa para convergencia desejada
    convergence_vector = pytorch_data['accuracy_training_per_epoch_vector']
    epoch = None
    for i in range(len(convergence_vector)):
        if convergence_vector[i]>=(convergencia/100):
            epoch=i+1
            break
    
    images_per_second_per_gb_memory = pytorch_data['images_per_second']/(pytorch_data['mean_peaky_gpu_memory']/1024)
    print("Training Data PyTorch x TensorFlow\n")

    print(100*"-")
    print(f"           Metrica             |            PyTorch           |            TensorFlow           ")
    print(100*"-")
    print(f"Tempo total de treinamento (s)     |       {pytorch_data['total_time_training']}      |                 -            ")
    print(f"Tempo médio por época (s) ± std    |       {pytorch_data['mean_time_per_epoch']} ± {pytorch_data['std_time_per_epoch']}      |                 -            ")
    print(f"Imagens por segundo     |       {pytorch_data['images_per_second']}       |                 -            ")
    print(f"Acurácia na última época (%)|        {pytorch_data['accuracy_training_per_epoch_vector'][-1]*100} |               -")
    print(f"Loss médio por época  ± std             |        {pytorch_data['mean_loss']} ± {pytorch_data['std_loss']} |               -")
    print(f"Épocas para {convergencia}%   de acurácia           |        {epoch} |               -")
    print(f"Tempo médio por batch (s)     |       {pytorch_data['time_per_batch']}       |                 -            ")
    print(f"Tempo para carregar os dados (s)     |       {pytorch_data['total_time_data_loading']}       |                 -            ")
    print(f"Tempo gasto para carregar os dados (%)     |       {pytorch_data['data_loading_ratio']*100}       |                 -            ")
    print(f"Tempo para computar o treino (s)     |       {pytorch_data['total_time_compute']}       |                 -            ")
    print(f"Tempo gasto para computar o treino (%)     |       {pytorch_data['compute_ratio']*100}       |                 -            ")
    print(f"Pico médio de uso de gpu (MB) ± std     | {pytorch_data['mean_peaky_gpu_memory']} ± {pytorch_data['std_peaky_gpu_memory']}  |                 -           ")
    print(f"Uso médio de gpu (%) ± std     | {pytorch_data['mean_usage_gpu_memory_percentage']*100} ± {pytorch_data['std_usage_gpu_memory']}  |                 -           ")
    print(f"Imagens/s/GB  | {images_per_second_per_gb_memory} |   -")

width_images = 3.3

def PlotAccuracyPerEpoch():
    pytorch_accuracy = pytorch_data['accuracy_training_per_epoch_vector']
    for i in range(len(pytorch_accuracy)):
        pytorch_accuracy[i] = pytorch_accuracy[i]*100

    plt.figure(figsize=(width_images*2,2*2))
    plt.plot(pytorch_accuracy, label="PyTorch", color='purple')
    plt.xlabel(r'$Época$')
    plt.ylabel(r'$Acurácia~(\%)$')
    plt.title(r'$Acurácia \,\,PyTorch \times \,\,TensorFlow$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()



def PlotLossPerEpoch():
    loss_epoch = pytorch_data['loss_per_epoch']
    plt.figure(figsize=(width_images*2,2*2))
    plt.plot(loss_epoch, label="PyTorch", color='purple')
    plt.xlabel(r'$Época$')
    plt.ylabel(r'$Loss$')
    plt.title(r'$Loss \,\,PyTorch \times \,\,TensorFlow$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()



ComparativeTable()
PlotAccuracyPerEpoch()
PlotLossPerEpoch()