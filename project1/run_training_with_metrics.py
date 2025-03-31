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


def get_gpu_metrics():
    """Collect GPU metrics using nvidia-smi."""
    try:
        output = subprocess.check_output(GPU_METRICS_CMD, shell=True).decode("utf-8").strip()
        if output:
            metrics = output.split(", ")
            gpu_util, mem_util, mem_used, mem_total = map(float, metrics)
            return {
                "gpu_utilization": gpu_util,
                "memory_utilization": mem_util,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
            }
    except subprocess.CalledProcessError:
        return None


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
        "python", "main.py",
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
