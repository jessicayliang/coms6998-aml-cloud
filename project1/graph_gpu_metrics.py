import json
import matplotlib.pyplot as plt

# File paths
resnet18_file = "gpu_metrics_logs/resnet18_gpu_metrics.json"
resnet50_file = "gpu_metrics_logs/resnet50_gpu_metrics.json"
alexnet_file = "gpu_metrics_logs/alexnet_gpu_metrics.json"

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines if line.strip()]

resnet18_data = load_json(resnet18_file)
resnet50_data = load_json(resnet50_file)
alexnet_data = load_json(alexnet_file)

# Extracting relevant metrics
def extract_metrics(data):
    gpu_util = [entry["gpu_utilization"] for entry in data]
    mem_util = [entry["memory_utilization"] for entry in data]
    flops_util = [entry.get("flops_utilization_tflops", 0) for entry in data]
    return gpu_util, mem_util, flops_util

resnet18_gpu, resnet18_mem, resnet18_flops = extract_metrics(resnet18_data)
resnet50_gpu, resnet50_mem, resnet50_flops = extract_metrics(resnet50_data)
alexnet_gpu, alexnet_mem, alexnet_flops = extract_metrics(alexnet_data)

# Plot GPU Utilization Comparison
plt.figure(figsize=(12, 6))
plt.plot(resnet18_gpu, label="ResNet18 GPU Util (%)", color='red', linestyle='--')
plt.plot(resnet50_gpu, label="ResNet50 GPU Util (%)", color='blue', linestyle='-.')
plt.plot(alexnet_gpu, label="AlexNet GPU Util (%)", color='green')
plt.xlabel("Time (steps)")
plt.ylabel("GPU Utilization (%)")
plt.title("GPU Utilization Comparison")
plt.legend()
plt.grid()
plt.show()

# Plot Memory Utilization Comparison
plt.figure(figsize=(12, 6))
plt.plot(resnet18_mem, label="ResNet18 Memory Util (%)", color='red', linestyle='--')
plt.plot(resnet50_mem, label="ResNet50 Memory Util (%)", color='blue', linestyle='-.')
plt.plot(alexnet_mem, label="AlexNet Memory Util (%)", color='green')
plt.xlabel("Time (steps)")
plt.ylabel("Memory Utilization (%)")
plt.title("Memory Utilization Comparison")
plt.legend()
plt.grid()
plt.show()

# Plot FLOPS Utilization Comparison
plt.figure(figsize=(12, 6))
plt.plot(resnet18_flops, label="ResNet18 FLOPS (TFLOPS)", color='red', linestyle='--')
plt.plot(resnet50_flops, label="ResNet50 FLOPS (TFLOPS)", color='blue', linestyle='-.')
plt.plot(alexnet_flops, label="AlexNet FLOPS (TFLOPS)", color='green')
plt.xlabel("Time (steps)")
plt.ylabel("FLOPS Utilization (TFLOPS)")
plt.title("FLOPS Utilization Comparison")
plt.legend()
plt.grid()
plt.show()
