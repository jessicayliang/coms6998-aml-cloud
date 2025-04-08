# Instructions

## Prerequisites
- Google Cloud SDK installed and configured
- kubectl installed and configured
- Access to a GCP project with GKE enabled
my commands:
```bash
gcloud auth login
gcloud config set project coms6998-spring2025
```

## Step 1: Create a GKE Cluster

```bash
# Set environment variables
export PROJECT_ID=$(gcloud config get-value project)
export CLUSTER_NAME=kubeflow-mnist-cluster
export ZONE=us-east1-b

# Create a GKE cluster
gcloud container clusters create $CLUSTER_NAME \
  --zone $ZONE \
  --machine-type n1-standard-4 \
  --num-nodes 2
  
# Configure kubectl to use the new cluster
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID
```

## docker repo in artifact (deprecated .io)
```bash
gcloud artifacts repositories create mnist-repo \
  --repository-format=docker \
  --location=us-east1 \
  --description="Repository for MNIST ML models"
  ```
```bash
gcloud auth configure-docker us-east1-docker.pkg.dev

```
## Step 3: Build and Push the Training Container

```bash


# Build and push the training container
docker buildx build --platform linux/amd64 -t us-east1-docker.pkg.dev/coms6698-spring2025/mnist-repo/mnist-training:latest -f Dockerfile .
docker push us-east1-docker.pkg.dev/coms6698-spring2025/mnist-repo/mnist-training:latest
```

## Step 4: Build and Push the Inference Container
docker buildx build --platform linux/amd64 -t us-east1-docker.pkg.dev/coms6698-spring2025/mnist-repo/mnist-inference:latest -f Dockerfile .
docker push us-east1-docker.pkg.dev/coms6698-spring2025/mnist-repo/mnist-inference:latest

## Step 5: Update YAML Files

Update all YAML files by replacing `[YOUR-PROJECT-ID]` with your actual GCP project ID.

## Step 6: Create the Persistent Volume Claim

```bash
cd ..
# Copy content from model-pvc.yaml to model-pvc.yaml

# Create the PVC
kubectl apply -f model-pvc.yaml
```

## Step 7: Run the Training Job

```bash

# Create the training job
kubectl apply -f training-job.yaml

# Monitor the training job
kubectl get jobs
kubectl logs -f job/mnist-training-job
```

## Step 8: Deploy the Inference Service

```bash
# Create the deployment and service
kubectl apply -f inference-deployment.yaml
kubectl apply -f inference-service.yaml

# Monitor the deployment
kubectl get deployments
kubectl get pods
```

## Step 9: Access the Inference Service

```bash
# Get the external IP of the service
kubectl get service mnist-inference-service

# Access the web interface using the EXTERNAL-IP
echo "Visit http://EXTERNAL-IP in your browser"
```

## Step 10: Test the Inference Service

1. Visit the web interface at http://EXTERNAL-IP
2. Use the file upload form to upload an image of a handwritten digit
3. Alternatively, use the `/predict/random` endpoint to test with a random MNIST digit

## Cleaning Up

```bash
# Delete the service and deployment
kubectl delete -f inference-service.yaml
kubectl delete -f inference-deployment.yaml

# Delete the training job
kubectl delete -f training-job.yaml

# Delete the PVC
kubectl delete -f model-pvc.yaml

# Delete the GKE cluster
gcloud container clusters delete $CLUSTER_NAME --zone $ZONE
```
