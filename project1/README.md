this project reports gpu utilization metrics and stores them in a separate json file.

clone this repository
download pytorch code for training imagenet data (https://github.com/pytorch/examples)
download https://image-net.org/ data (only a small subset)

in the project1 repository, run: python run_training_with_metrics.py --data-path ../../examples/imagenet/tiny-imagenet-200/

"../../examples/imagenet/tiny-imagenet-200/" is the datapath to the downloaded dataset (and most likely yours if you followed this, but you should double check).

I only pasted it here for my personal use (I am working close to the deadline).

training will automatically run for resnet18, resnet50, and alexnet with metrics stores in a new directory, "gpu_metrics_logs."
