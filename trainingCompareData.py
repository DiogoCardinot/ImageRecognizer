import os 
import json
import matplotlib.pyplot as plt

training_pytorch_data = "./pytorch/trainingMetrics/pytorch_metrics.json"

with open(training_pytorch_data, mode='r') as file:
    pytorch_data = json.load(file)


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
    loss_epoch_pytorch = pytorch_data['loss_per_epoch']
    plt.figure(figsize=(width_images*2,2*2))
    plt.plot(loss_epoch_pytorch, label="PyTorch", color='purple')
    plt.xlabel(r'$Época$')
    plt.ylabel(r'$Loss$')
    plt.title(r'$Loss \,\,PyTorch \times \,\,TensorFlow$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def PlotTimePerEpoch():
    time_epoch_pytorch = pytorch_data['time_per_epoch']
    plt.figure(figsize=(width_images*2,2*2))
    plt.plot(time_epoch_pytorch, label="PyTorch", color='purple')
    plt.xlabel(r'$Época$')
    plt.ylabel(r'$Tempo~(s)$')
    plt.title(r'$Time \,\,PyTorch \times \,\,TensorFlow$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def PlotDataLoadingAndComputePerEpoch():
    data_loading_time_pytorch = pytorch_data['total_batch_data_loading_time_per_epoch']
    compute_time_pytorch = pytorch_data['total_batch_compute_per_epoch']

    x = [i for i in range(len(data_loading_time_pytorch))]

    fig, ax = plt.subplots(2, 1, figsize=(width_images*2,2*2*2))
    ax[0].text(-0.15, 1.12, '(a)', transform=ax[0].transAxes, va='top', fontsize = 15)
    ax[0].bar(x, data_loading_time_pytorch, color = 'purple', label = 'Data Loading')
    ax[0].bar(x, compute_time_pytorch, bottom = data_loading_time_pytorch , color = 'black', label='Compute Time')
    ax[1].text(-0.15, 1.15, '(b)', transform=ax[1].transAxes, va='top', fontsize = 15)
    ax[1].bar(x, data_loading_time_pytorch, color = 'blue', label = 'Data Loading')
    ax[1].bar(x, compute_time_pytorch, bottom = data_loading_time_pytorch , color = 'orange', label='Compute Time')

    ax[0].set_xlabel(r'$Época$')
    ax[0].set_ylabel(r'$Compute \,\, time \times \,\, Data \,\, loading \,\, time~(s)$')

    ax[1].set_xlabel(r'$Época$')
    ax[1].set_ylabel(r'$Compute \,\, time \times \,\, Data \,\, loading \,\, time~(s)$')

    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    plt.tight_layout()
    plt.show()

def PlotGPUMemoryPerEpoch():
    gpu_memory_per_epoch_pytorch = pytorch_data['peaky_gpu_memory_per_epoch']

    plt.figure(figsize=(width_images*2,2*2))
    plt.plot(gpu_memory_per_epoch_pytorch, label="PyTorch", color='purple')
    plt.xlabel(r'$Época$')
    plt.ylabel(r'$Peaky \,\, GPU \,\, memory \,\, usage$')
    plt.title(r'$Peaky \,\, GPU \,\, usage \,\,PyTorch \times \,\,TensorFlow$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def PlotTimePerBatchOneEpoch(epoch = 50):
    data_loading_batch_time_epoch_pytorch = pytorch_data['data_loading_batch_times'][f'{epoch}']
    compute_batch_time_epoch_pytorch = pytorch_data['compute_batch_times'][f'{epoch}']

    fig, ax = plt.subplots(2,1, figsize = (width_images*2,2*2*2), sharey=True)

    # ymin = min(min(data_loading_batch_time_epoch_pytorch), min(compute_pt),
    #        min(data_loading_tf), min(compute_tf))
    # ymax = max(max(data_loading_batch_time_epoch_pytorch), max(compute_batch_time_epoch_pytorch),
    #         max(data_loading_tf), max(compute_tf))

    #Completar com o min e max do tensorflow
    ymin = min(min(data_loading_batch_time_epoch_pytorch), min(compute_batch_time_epoch_pytorch))
    ymax = max(max(data_loading_batch_time_epoch_pytorch), max(compute_batch_time_epoch_pytorch))

    ax[0].text(-0.15, 1.12, '(a)', transform=ax[0].transAxes, va='top', fontsize = 15)
    ax[0].boxplot([data_loading_batch_time_epoch_pytorch, compute_batch_time_epoch_pytorch], labels=["Data loading time", "Compute time"], vert=True)
    ax[0].set_ylabel("Time (s)")

    ax[1].text(-0.15, 1.15, '(b)', transform=ax[1].transAxes, va='top', fontsize = 15)
    ax[1].boxplot([data_loading_batch_time_epoch_pytorch, compute_batch_time_epoch_pytorch], labels=["Data loading time", "Compute time"], vert=True)
    ax[1].set_ylabel("Time (s)")

    ax[0].set_ylim(ymin-0.2*(ymax/10), ymax+0.1*ymax)
    ax[1].set_ylim(ymin-0.2*(ymax/10), ymax+0.1*ymax)
    plt.tight_layout()
    plt.show()
    
def PlotImagesPerSecondPerEpoch():
    # Throughput ao longo do treinamento
    images_per_second_epoch_pytorch = pytorch_data['images_per_second_epoch']

    plt.figure(figsize=(width_images*2,2*2))
    plt.plot(images_per_second_epoch_pytorch, label="PyTorch", color='purple')
    plt.xlabel(r'$Época$')
    plt.ylabel(r'$Images/s$')
    plt.title(r'$Images \,\, per \,\, second \,\, per \,\, epoch \,\,PyTorch \times \,\,TensorFlow$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


ComparativeTable()
PlotAccuracyPerEpoch()
PlotLossPerEpoch()
PlotDataLoadingAndComputePerEpoch()
PlotTimePerEpoch()
PlotGPUMemoryPerEpoch()
PlotImagesPerSecondPerEpoch()
PlotTimePerBatchOneEpoch()