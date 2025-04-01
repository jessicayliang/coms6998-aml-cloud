import json

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def get_max_metrics(data):
    # Extract values safely using .get()
    gpu_utilization = [entry.get('gpu_utilization', 0) for entry in data]
    memory_bandwidth = [entry.get('memory_bandwidth_mb_s', 0) for entry in data if entry.get('memory_bandwidth_mb_s') is not None]
    flop_utilization = [entry.get('flops_utilization_tflops', 0) for entry in data]

    # Ensure we don't call max() on an empty list
    return {
        'max_gpu_utilization': max(gpu_utilization) if gpu_utilization else None,
        'max_memory_bandwidth_mb_s': max(memory_bandwidth) if memory_bandwidth else None,
        'max_flops_utilization_tflops': max(flop_utilization) if flop_utilization else None
    }

# Load GPU metrics for each model
container_alexnet_data = load_json('container_alexnet_gpu_metrics.json')
vm_alexnet_data = load_json('vm_alexnet_gpu_metrics.json')

container_resnet18_data = load_json('container_resnet18_gpu_metrics.json')
vm_resnet18_data = load_json('vm_resnet18_gpu_metrics.json')

container_resnet50_data = load_json('container_resnet50_gpu_metrics.json')
vm_resnet50_data = load_json('vm_resnet50_gpu_metrics.json')

# Get max values
models = {
    'Container AlexNet': container_alexnet_data,
    'VM AlexNet': vm_alexnet_data,
    'Container ResNet18': container_resnet18_data,
    'VM ResNet18': vm_resnet18_data,
    'Container ResNet50': container_resnet50_data,
    'VM ResNet50': vm_resnet50_data
}

for model, data in models.items():
    if not data:
        print(f"Warning: No data found for {model}!")
        continue

    max_metrics = get_max_metrics(data)
    print(f"{model} Max Metrics: {max_metrics}")
