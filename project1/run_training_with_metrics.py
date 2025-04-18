import os
import subprocess
import argparse
import time
import json
import re

# ============================
# CONFIGURATION
# ============================
GPU_METRICS_CMD = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits"
FLOPS_CMD = "nvidia-smi dmon -s u"  # Basic utilization, better with nvprof for detailed profiling
METRICS_LOG_DIR = "gpu_metrics_logs"

# Create the logs directory
os.makedirs(METRICS_LOG_DIR, exist_ok=True)

def get_memory_bandwidth():
    """Get GPU memory bandwidth using nvidia-smi nvlink."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pcie.link.gen.max,pcie.link.width.max',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True
        )
        
        # Parse output
        values = result.stdout.strip().split(', ')
        if len(values) == 2:
            link_gen = int(values[0])  # PCIe Gen
            link_width = int(values[1])  # Number of PCIe lanes
            
            # PCIe Gen max bandwidth per lane (GB/s)
            pcie_bandwidth_table = {
                1: 0.25,  # PCIe Gen 1: 250 MB/s per lane
                2: 0.5,   # PCIe Gen 2: 500 MB/s per lane
                3: 0.985, # PCIe Gen 3: ~985 MB/s per lane
                4: 1.969, # PCIe Gen 4: ~1969 MB/s per lane
                5: 3.938  # PCIe Gen 5: ~3938 MB/s per lane
            }
            
            # Estimate memory bandwidth
            mem_bandwidth = link_width * pcie_bandwidth_table.get(link_gen, 0)
            return mem_bandwidth * 1000  # Convert to MB/s
        return 0.0
    except Exception as e:
        print(f"Error getting memory bandwidth: {e}")
        return 0.0

def get_num_cores():
    """Get the number of cores per SM dynamically based on GPU type."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stdout=subprocess.PIPE, text=True
        )

        gpu_name = result.stdout.strip()

        # Define known GPUs and their cores per SM
        core_mapping = {
            "Tesla T4": 1280,
            "A100-SXM4-40GB": 6912,
            "V100-SXM2-16GB": 5120,
            "L4": 7680
        }

        return core_mapping.get(gpu_name, 1280)  # Default to T4 if unknown
    except Exception as e:
        print(f"Error detecting GPU name: {e}")
        return 1280

def estimate_flops(gpu_utilization, gpu_clock_mhz, num_cores, ops_per_cycle=2):
    """Estimate FLOPS using basic approximation formula."""
    return gpu_utilization / 100 * gpu_clock_mhz * 1e6 * num_cores * ops_per_cycle

def get_gpu_metrics():
    """Collect GPU metrics including utilization, memory, bandwidth, and FLOPS."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,clocks.sm',
                             '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, text=True)
    
    # Split result by commas and parse
    values = result.stdout.strip().split(', ')
    if len(values) == 4:
        gpu_utilization = float(values[0])
        memory_used = float(values[1])
        memory_total = float(values[2])
        gpu_clock_mhz = float(values[3])

        # Get memory bandwidth
        mem_bandwidth = get_memory_bandwidth()

        # Estimate FLOPS (if needed)
        num_cores = get_num_cores()  # Example for NVIDIA T4, adjust based on GPU type
        flops_utilization = estimate_flops(gpu_utilization, gpu_clock_mhz, num_cores)

        # Return all metrics
        return {
            "gpu_utilization": gpu_utilization,
            "memory_utilization": round((memory_used / memory_total) * 100, 2) if memory_total > 0 else 0,
            "memory_used_mb": memory_used,
            "memory_total_mb": memory_total,
            "memory_bandwidth_mb_s": mem_bandwidth,
            "flops_utilization_tflops": flops_utilization / 1e12  # Convert to TFLOPS
        }
    else:
        return {"gpu_utilization": 0.0, "memory_utilization": 0.0, "memory_used_mb": 0.0,
                "memory_total_mb": 0.0, "memory_bandwidth_mb_s": 0.0, "flops_utilization_tflops": 0.0}

def log_metrics(log_file, metrics):
    """Write GPU metrics to the log file."""
    with open(log_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def run_model(model_name, data_path, epochs=90, batch_size=256, patience=5, gpu_threshold=95.0):
    """
    Run the training command and monitor hardware performance.

    Args:
    - model_name (str): Model to run (resnet18, resnet50, alexnet)
    - data_path (str): Path to ImageNet/Tiny ImageNet dataset
    - epochs (int): Maximum epochs to run
    - batch_size (int): Batch size for training
    - patience (int): Epochs to wait before stopping if GPU utilization plateaus
    - gpu_threshold (float): Threshold for sustained GPU utilization to trigger early stopping
    """

    # Define training command
    cmd = [
        "python3", "/app/pytorch-examples/imagenet/main.py",
        "-a", model_name,  # Model architecture
        data_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--print-freq", "10"
    ]

    log_file = f"{METRICS_LOG_DIR}/{model_name}_gpu_metrics.json"
    log_file_txt = f"{METRICS_LOG_DIR}/{model_name}_log.txt"

    print(f"\n Starting training for {model_name}...")

    # Run the training process
    with open(log_file_txt, "w") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        best_gpu_util = 0
        plateau_count = 0

        for line in iter(process.stdout.readline, ""):
            f.write(line)
            f.flush()
            print(line, end="")

            # Get current GPU metrics
            metrics = get_gpu_metrics()
            if metrics:
                log_metrics(log_file, metrics)

                # Monitor GPU utilization for early stopping
                gpu_util = metrics["gpu_utilization"]
                if gpu_util > best_gpu_util:
                    best_gpu_util = gpu_util
                    plateau_count = 0
                elif gpu_util < best_gpu_util - 5.0:  # Small tolerance for fluctuation
                    plateau_count += 1

            # Stop training if GPU utilization plateaus for `patience` epochs
            if plateau_count >= patience:
                print(f"\n Early stopping: GPU utilization plateaued at {best_gpu_util}%")
                process.terminate()
                break

    # Save the collected metrics
    print(f"GPU metrics saved to {log_file}")
    process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run and monitor model training with GPU metrics.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "resnet50", "alexnet"],
        help="List of models to train (default: resnet18, resnet50, alexnet)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to ImageNet/Tiny ImageNet dataset with 'train' and 'val' folders"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=90,
        help="Number of epochs to train (default: 90)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs before stopping if GPU utilization plateaus (default: 5)"
    )
    parser.add_argument(
        "--gpu-threshold",
        type=float,
        default=95.0,
        help="GPU utilization threshold for stopping (default: 95%)"
    )

    args = parser.parse_args()

    # Run training for each model
    for model_name in args.models:
        print(f"\n Starting training with {model_name}...")
        run_model(
            model_name,
            args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            gpu_threshold=args.gpu_threshold
        )


if __name__ == "__main__":
    main()
