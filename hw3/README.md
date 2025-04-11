# how to use run inference service with web interface (mnist)

## prereqs
- Google cloud SDK installed and configured
- kubectl installed and configured
- Access to a GCP project with GKE enabled

### my commands for my project
```bash
gcloud auth login
gcloud config set project coms6998-spring2025
```

## Create GKE cluster

```bash
# set env variables
export PROJECT_ID=$(gcloud config get-value project)
export CLUSTER_NAME=kubeflow-mnist-cluster
export ZONE=us-east1-b

# create cluster
gcloud container clusters create $CLUSTER_NAME \
  --zone $ZONE \
  --machine-type n1-standard-4 \
  --num-nodes 2
  
# Configure kubectl to use the new cluster
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID
```

## docker repo in artifact (since writing to container registry deprecated 03/18/2025)
https://cloud.google.com/container-registry/docs/deprecations/container-registry-deprecation
```bash
gcloud artifacts repositories create mnist-repo \
  --repository-format=docker \
  --location=us-east1 \
  --description="Repository for MNIST"
  ```

### double check and list artifact repos - should see mnist-repo
```bash
gcloud artifacts repositories list
```

## configure docker
```bash
gcloud auth configure-docker us-east1-docker.pkg.dev

```

## build and push training container to artifact repo for amd64
remember to go to training directory
```bash
docker buildx build --platform linux/amd64 \
  -t us-east1-docker.pkg.dev/coms6998-spring2025/mnist-repo/mnist-training:latest \
  --push \
  -f Dockerfile .
```
### double check and list images
```bash
gcloud artifacts docker images list us-east1-docker.pkg.dev/coms6998-spring2025/mnist-repo

```
#### in case something goes wrong and need to delete images
```bash
gcloud artifacts docker images delete \
  us-east1-docker.pkg.dev/coms6998-spring2025/mnist-repo/mnist-training \
  --delete-tags \
  --quiet
```

## build and push inference container (same as training)
remember to go to inference directory
```bash
docker buildx build --platform linux/amd64 \
-t us-east1-docker.pkg.dev/coms6998-spring2025/mnist-repo/mnist-inference:latest \
--push \
-f Dockerfile .
```

### double check and list images
```bash
gcloud artifacts docker images list us-east1-docker.pkg.dev/coms6998-spring2025/mnist-repo
```

#### in case you need to delete inference images
```bash
 gcloud artifacts docker images delete \
  us-east1-docker.pkg.dev/coms6998-spring2025/mnist-repo/mnist-inference \
  --delete-tags \
  --quiet
```

## Create persistent volume claim
go to k8s directory

```bash
kubectl apply -f model-pvc.yaml
```

## run training
```bash
kubectl apply -f training-job.yaml
```

### monitoring jobs
```bash
kubectl get jobs
kubectl logs -f job/mnist-training-job
```

## deploy inference services

```bash
# creating
kubectl apply -f inference-deployment.yaml
kubectl apply -f inference-service.yaml
```

### monitoring
```bash
kubectl get deployments
kubectl get pods
```

## access the inferences service

```bash
# get external ip
kubectl get service mnist-inference-service
```

## test inference service

1. web interface at http://EXTERNAL-IP

## clean up (image deletion commands above)
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
