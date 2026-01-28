import json
import matplotlib.pyplot as plt

test_pytorch_path = "./pytorch/testMetrics/pytorch_metrics.json"

with open(test_pytorch_path, 'r') as file:
    pytorch_test_data = json.load(file)

def ComparativeTable():
    images_per_second_per_gb_memory = pytorch_test_data['images_per_second']/(pytorch_test_data['peaky_gpu_memory']/1024)
    print("Test Data PyTorch x TensorFlow\n")
    print(100*"-")
    print(f"           Metrica             |            PyTorch           |            TensorFlow           ")
    print(100*"-")
    print(f"Tempo total de teste (s)     |       {pytorch_test_data['total_time_test']}      |                 -            ")
    print(f"Imagens por segundo     |       {pytorch_test_data['images_per_second']}       |                 -            ")
    print(f"Acurácia final (%)|        {pytorch_test_data['final_accuracy']} |               -")
    print(f"Tempo médio por batch (s) ± std    |       {pytorch_test_data['mean_time_per_batch']} ± {pytorch_test_data['std_time_per_batch']}       |                 -            ")
    print(f"Tempo total de carregamento de dados (s)     |       {pytorch_test_data['total_data_loading_time']}       |                 -            ")
    print(f"Proporção do tempo em carregamento (%)     |       {pytorch_test_data['data_loading_ratio']*100}       |                 -            ")
    print(f"Tempo total de inferência (s)    |       {pytorch_test_data['total_compute_time']}       |                 -            ")
    print(f"Proporção do tempo em inferência (%)     |       {pytorch_test_data['compute_ratio']*100}       |                 -            ")
    print(f"Pico de uso de memória GPU (MB)   | {pytorch_test_data['peaky_gpu_memory']} |                 -           ")
    print(f"Uso máximo de memória GPU (%)   | {pytorch_test_data['peaky_peaky_gpu_memory_percentage']*100}   |                 -           ")
    print(f"Imagens por segundo por GB de memória  | {images_per_second_per_gb_memory} |   -")

width_fig = 3.3

def PlotComputeDataLoadingTimes():
    compute_test_time_pytorch = pytorch_test_data['batch_compute_times']
    data_loading_test_time_pytorch = pytorch_test_data['batch_data_loading_times']

    fig, ax = plt.subplots(2,1, figsize=(width_fig*2,2*2*2))

    # ymin = min(min(data_loading_test_time_pytorch), min(compute_test_time_pytorch),
    #        min(data_loading_tf), min(compute_tf))
    # ymax = max(max(data_loading_test_time_pytorch), max(compute_test_time_pytorch),
    #         max(data_loading_tf), max(compute_tf))

    #Completar com o min e max do tensorflow
    ymin = min(min(data_loading_test_time_pytorch), min(compute_test_time_pytorch))
    ymax = max(max(data_loading_test_time_pytorch), max(compute_test_time_pytorch))

    ax[0].text(-0.15, 1.12, '(a)', transform=ax[0].transAxes, va='top', fontsize = 15)
    ax[0].boxplot([data_loading_test_time_pytorch, compute_test_time_pytorch], labels=['Data loading', 'Compute'], vert=True)
    ax[0].set_ylabel("Time (s)")

    ax[1].text(-0.15, 1.12, '(b)', transform=ax[1].transAxes, va='top', fontsize = 15)
    ax[1].boxplot([data_loading_test_time_pytorch, compute_test_time_pytorch], labels=['Data loading', 'Compute'], vert=True)
    ax[1].set_ylabel("Time (s)")

    ax[0].set_ylim(ymin-0.2*(ymax/10), ymax+0.1*ymax)
    ax[1].set_ylim(ymin-0.2*(ymax/10), ymax+0.1*ymax)
    plt.tight_layout()
    plt.show()

# ComparativeTable()
PlotComputeDataLoadingTimes()