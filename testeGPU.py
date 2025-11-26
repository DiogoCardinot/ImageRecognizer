# teste_gpu_completo.py

# https://www.youtube.com/watch?v=ryFFCyhTgyA

import tensorflow as tf
import os

print("=== TESTE COMPLETO GPU - TensorFlow 2.10 ===")

# 1. Configurar para mostrar todos os logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# 2. Verificar versões
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {tf.__version__.split(' ')[0]}")

# 3. Listar TODOS os dispositivos físicos
print("\n=== DISPOSITIVOS FÍSICOS ===")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(f"  {device}")

# 4. Verificar GPUs especificamente
print("\n=== GPUs ESPECÍFICAS ===")
gpus = tf.config.list_physical_devices('GPU')
print(f"Número de GPUs disponíveis: {len(gpus)}")

if gpus:
    print("🎉 GPU DETECTADA! Configurando...")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        # Configurar memory growth para evitar usar toda VRAM de uma vez
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth habilitado")
else:
    print("Nenhuma GPU detectada")
    exit()

# 5. Teste de PERFORMANCE na GPU
print("\n=== TESTE DE PERFORMANCE ===")
try:
    # Forçar operação na GPU
    with tf.device('/GPU:0'):
        print("Executando operação matricial na GPU...")
        
        # Criar tensores grandes
        a = tf.random.normal([10000, 1000])
        b = tf.random.normal([1000, 5000])
        
        print(f"  Tensor A: {a.shape}")
        print(f"  Tensor B: {b.shape}")
        
        # Operação matricial (muito pesada para CPU)
        c = tf.matmul(a, b)
        
        print(f"  Resultado: {c.shape}")
        print("Operação matricial concluída na GPU!")
        
        # Teste simples de cálculo
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        y = tf.constant([2.0, 2.0, 2.0, 2.0, 2.0])
        z = x * y
        
        print(f"  Teste simples: {z.numpy()}")
        
except Exception as e:
    print(f"Erro durante operação GPU: {e}")

# 6. Verificar memória GPU
print("\n=== INFORMAÇÕES DE MEMÓRIA ===")
if gpus:
    try:
        # Tentar obter informações de memória (pode não funcionar em todas versões)
        from tensorflow.python.client import device_lib
        
        local_device_protos = device_lib.list_local_devices()
        for device in local_device_protos:
            if device.device_type == 'GPU':
                print(f"  GPU: {device.physical_device_desc}")
                print(f"  Memória: {device.memory_limit / 1024**3:.1f} GB")
    except:
        print("  Informações detalhadas não disponíveis")

print("\n=== RESULTADO FINAL ===")
if gpus:
    print("SUA GPU ESTÁ FUNCIONANDO PERFEITAMENTE COM TENSORFLOW 2.10! 🎊")
    print("Pronto para treinar suas CNNs com alta performance!")
else:
    print("Ainda há problemas na configuração da GPU")

print("\nDica: Execute sua CNN e observe a velocidade - deve ser MUITO mais rápida! 🚀")