import json
import matplotlib.pyplot as plt
import numpy as np  # For generating time values

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

# Function to extract specific metric data
def extract_metric(data, metric):
    return [entry.get(metric, np.nan) for entry in data]

# Load GPU metrics for each model from the corresponding files
models = ['alexnet', 'resnet18', 'resnet50']
metrics = ['gpu_utilization', 'memory_utilization', 'memory_bandwidth_mb_s', 'flops_utilization_tflops']

data = {}
for model in models:
    data[f'container_{model}'] = load_json(f'container_{model}_gpu_metrics.json')
    data[f'vm_{model}'] = load_json(f'vm_{model}_gpu_metrics.json')

# Extract data for each metric
data_extracted = {}
for key, dataset in data.items():
    data_extracted[key] = {metric: extract_metric(dataset, metric) for metric in metrics}

# Ensure that time axis length accommodates both datasets
time_axes = {}
max_lengths = {model: max(len(data_extracted[f'container_{model}']['gpu_utilization']),
                          len(data_extracted[f'vm_{model}']['gpu_utilization'])) for model in models}

time_axes = {model: np.arange(max_lengths[model]) for model in models}

# Pad shorter datasets with NaN values
for model in models:
    for key in ['container', 'vm']:
        for metric in metrics:
            data_extracted[f'{key}_{model}'][metric] = np.pad(
                data_extracted[f'{key}_{model}'][metric],
                (0, max_lengths[model] - len(data_extracted[f'{key}_{model}'][metric])),
                'constant',
                constant_values=np.nan
            )

# Create subplots for the metrics comparison
fig, axes = plt.subplots(len(models), len(metrics), figsize=(20, 15))

# Plot each metric for each model
for i, model in enumerate(models):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        ax.plot(time_axes[model], data_extracted[f'container_{model}'][metric], label=f"Container {model.capitalize()}", color='blue')
        ax.plot(time_axes[model], data_extracted[f'vm_{model}'][metric], label=f"VM {model.capitalize()}", color='red')
        ax.set_title(f'{model.capitalize()} {metric.replace("_", " ").capitalize()} Comparison')
        ax.set_xlabel('Time')
        ax.set_ylabel(metric.replace("_", " ").capitalize())
        ax.legend()

# Adjust layout for better readability
plt.tight_layout()

# Save the plot as an image
plt.savefig('gpu_metrics_comparison.png')

# Show the plot
plt.show()