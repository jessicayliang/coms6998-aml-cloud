import time
import argparse
import logging
import uuid
from typing import Optional, Tuple

from google.cloud import compute_v1
from google.api_core.exceptions import GoogleAPIError, PermissionDenied, ResourceExhausted, InvalidArgument

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of supported GPU types
GPU_TYPES = [
    "nvidia-tesla-t4",
    "nvidia-l4",
    "nvidia-tesla-v100",
    "nvidia-tesla-p100",
    "nvidia-tesla-p4",
    "nvidia-tesla-a100",
    "nvidia-a100-80gb",
]

# VM configurations for each GPU type
VM_CONFIGS = {
    "nvidia-tesla-t4": {"machine_type": "n1-standard-4"},
    "nvidia-tesla-v100": {"machine_type": "n1-standard-8"},
    "nvidia-tesla-p100": {"machine_type": "n1-standard-8"},
    "nvidia-tesla-p4": {"machine_type": "n1-standard-8"},
    "nvidia-tesla-a100": {"machine_type": "a2-highgpu-1g"},
    "nvidia-a100-80gb": {"machine_type": "a2-ultragpu-1g"},
    "nvidia-l4": {"machine_type": "g2-standard-8"}
}

# Priority regions to check first
PRIORITY_REGIONS = ["us-central1", "us-east1", "us-west1", "europe-west4", "asia-east1"]

class GPUFinder:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient()
        self.zones = compute_v1.ZonesClient()
        self.images = compute_v1.ImagesClient()
        self.latest_img = self._get_latest_img()
        self.all_zones = self._get_all_zones()
        self.zone_to_region = self._get_zone_region_mapping()

    def _get_all_zones(self):
        """Get all available zones in the project"""
        try:
            request = compute_v1.ListZonesRequest(project=self.project_id)
            zones = self.zones.list(request=request)
            return [zone.name for zone in zones]
        except GoogleAPIError as e:
            logger.error(f"Failed to get zones: {e}")
            return []

    def _get_zone_region_mapping(self):
        """Create a mapping of zones to their regions"""
        zone_to_region = {}
        for zone in self.all_zones:
            # e.g., us-central1-a to us-central1
            region = "-".join(zone.split("-")[:-1])
            zone_to_region[zone] = region
        return zone_to_region

    def _get_latest_img(self):
        """Get the latest deep learning image with GPU support"""
        try:
            request = compute_v1.ListImagesRequest(
                project="deeplearning-platform-release",
                filter="family=tf-latest-gpu"
            )
            images = self.images.list(request=request)
            image_list = list(images)
            if image_list:
                latest_image = image_list[0]
                return f"projects/deeplearning-platform-release/global/images/{latest_image.name}"
            else:
                return "projects/deeplearning-platform-release/global/images/family/tf-latest-gpu"
        except GoogleAPIError as e:
            logger.warning(f"Failed to get latest deep learning image: {e}. Using fallback.")
            return "projects/deeplearning-platform-release/global/images/family/tf-latest-gpu"

    def try_create_vm(self, zone: str, gpu_type: str) -> Tuple[bool, Optional[str]]:
        """Attempt to create a VM with the specified GPU type in the specified zone"""
        instance_name = f"gpu-test-{uuid.uuid4().hex[:8]}"

        try:
            vm_config = VM_CONFIGS.get(gpu_type, VM_CONFIGS["nvidia-tesla-t4"])
            machine_type = vm_config["machine_type"]

            # Create instance configuration
            instance = compute_v1.Instance()
            instance.name = instance_name
            instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

            # Configure boot disk
            disk = compute_v1.AttachedDisk()
            initialize_params = compute_v1.AttachedDiskInitializeParams()
            initialize_params.disk_size_gb = 100
            initialize_params.source_image = self.latest_img
            disk.initialize_params = initialize_params
            disk.auto_delete = True
            disk.boot = True
            instance.disks = [disk]

            # Configure network interface
            network_interface = compute_v1.NetworkInterface()
            access_config = compute_v1.AccessConfig()
            network_interface.access_configs = [access_config]
            instance.network_interfaces = [network_interface]

            # Configure scheduling (preemptible to save cost)
            scheduling = compute_v1.Scheduling()
            scheduling.preemptible = True
            scheduling.automatic_restart = False
            instance.scheduling = scheduling

            # Configure GPU
            guest_accelerator = compute_v1.AcceleratorConfig()
            guest_accelerator.accelerator_count = 1
            guest_accelerator.accelerator_type = f"projects/{self.project_id}/zones/{zone}/acceleratorTypes/{gpu_type}"
            instance.guest_accelerators = [guest_accelerator]

            # Create the VM
            request = compute_v1.InsertInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance_resource=instance,
            )
            operation = self.compute_client.insert(request=request, timeout=30)
            logger.info(f"Success - Created VM {instance_name} with {gpu_type} GPU in {zone}")

            # Delete the VM to avoid charges
            delete_request = compute_v1.DeleteInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance=instance.name
            )
            self.compute_client.delete(request=delete_request)
            logger.info(f"Deleted VM {instance_name}")

            return True, f"Successfully created a VM with {gpu_type} GPU in {zone}"

        except (PermissionDenied, ResourceExhausted, InvalidArgument, GoogleAPIError) as e:
            error_msg = str(e)
            logger.warning(f"Failed to create VM in {zone} with {gpu_type}: {error_msg}")
            return False, error_msg

    def find_gpu_vm(self):
        """Find a zone and GPU type where a VM can be created"""
        # First check priority zones
        priority_zones = [zone for zone in self.all_zones
                          if self.zone_to_region.get(zone) in PRIORITY_REGIONS]

        logger.info(f"Checking {len(priority_zones)} priority zones first...")
        for zone in priority_zones:
            for gpu_type in GPU_TYPES:
                logger.info(f"Trying {gpu_type} in {zone}...")
                success, message = self.try_create_vm(zone, gpu_type)
                if success:
                    return zone, gpu_type, message

        # If no success in priority zones, check remaining zones
        remaining_zones = [zone for zone in self.all_zones if zone not in priority_zones]
        logger.info(f"Checking {len(remaining_zones)} remaining zones...")
        for zone in remaining_zones:
            for gpu_type in GPU_TYPES:
                logger.info(f"Trying {gpu_type} in {zone}...")
                success, message = self.try_create_vm(zone, gpu_type)
                if success:
                    return zone, gpu_type, message

        return None, None, "No GPU VMs available in any zone"

def main():
    parser = argparse.ArgumentParser(description='Find a zone with GPU availability')
    parser.add_argument('--project', type=str, required=True, help='GCP Project ID')
    args = parser.parse_args()

    finder = GPUFinder(project_id=args.project)

    logger.info("Starting GPU VM search...")
    zone, gpu_type, message = finder.find_gpu_vm()

    if zone and gpu_type:
        print(f"\n SUCCESS: {message}")
        print(f"\nTo create a VM with this GPU, use:")
        print(f"gcloud compute instances create my-gpu-instance \\")
        print(f"  --project={args.project} \\")
        print(f"  --zone={zone} \\")
        print(f"  --machine-type={VM_CONFIGS[gpu_type]['machine_type']} \\")
        print(f"  --accelerator=type={gpu_type},count=1 \\")
        print(f"  --image-family=tf-latest-gpu \\")
        print(f"  --image-project=deeplearning-platform-release \\")
        print(f"  --boot-disk-size=100GB \\")
        print(f"  --maintenance-policy=TERMINATE")
    else:
        print(f"\n FAILED: {message}")

if __name__ == "__main__":
    main()
