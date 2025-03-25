import time
import concurrent.futures
import argparse
import logging
import uuid
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import requests
from google.cloud import compute_v1
from google.api_core.exceptions import GoogleAPIError, PermissionDenied, ResourceExhausted, InvalidArgument
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GPU_TYPES = ["nvidia-tesla-t4","nvidia-l4","nvidia-tesla-v100","nvidia-tesla-p100","nvidia-tesla-p4","nvidia-tesla-a100","nvidia-a100-80gb",
]

VM_CONFIGS = {
    "nvidia-tesla-t4": {"machine_type": "n1-standard-4"},
    "nvidia-tesla-v100": {"machine_type": "n1-standard-8"},
    "nvidia-tesla-p100": {"machine_type": "n1-standard-8"},
    "nvidia-tesla-p4": {"machine_type": "n1-standard-8"},
    "nvidia-tesla-a100": {"machine_type": "a2-highgpu-1g"},
    "nvidia-a100-80gb": {"machine_type": "a2-ultragpu-1g"},
    "nvidia-l4": {"machine_type": "g2-standard-8"}
}

@dataclass
class Result:
    zone: str
    gpu_type: str
    is_available: bool
    method: str
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    latency: float = 0.0

class Checker:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient()
        self.regions = compute_v1.RegionsClient()
        self.zones = compute_v1.ZonesClient()
        self.gpu_types = compute_v1.AcceleratorTypesClient()
        self.images = compute_v1.ImagesClient()

        # Caches
        self.gpu_cache: Dict[str, List[str]] = {}
        self.gpu_quotas_cache: Dict[str, Dict[str, float]] = {}

        self.successes: List[Result] = []
        self.performance: Dict[str, List[float]] = {
            "list_accelerator_types": [],
            "check_quota": [],
            "create_and_delete_vm": []
        }
        self.all_zones = self._get_all_zones()
        self.zone_to_region = self._get_zone_region_mapping()
        self.latest_img = self._get_latest_img()

    def _get_all_zones(self) -> List[str]:
        try:
            request = compute_v1.ListZonesRequest(project=self.project_id)
            zones = self.zones.list(request=request)
            zone_names = [zone.name for zone in zones]
            return zone_names
        except GoogleAPIError as e:
            logger.error(f"failed get zones: {e}")
            return []

    def _get_zone_region_mapping(self) -> Dict[str, str]:
        zone_to_region = {}
        for zone in self.all_zones:
            # us-central1-a to us-central1)
            region = "-".join(zone.split("-")[:-1])
            zone_to_region[zone] = region
        return zone_to_region

    def _get_latest_img(self) -> str:
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

    def get_all_zones(self) -> List[str]:
        return self.all_zones

    def check_permissions(self):
        try:
            zones = self.zones.list(request=compute_v1.ListZonesRequest(project=self.project_id))
            return True
        except PermissionDenied as e:
            logger.error(f"permission error: {e}")
            return False

    def check_gpu_zone(self, zone: str) -> List[str]:
        """
        Check which (GPUs) available in a zone
        """
        start_time = time.time()
        try:
            # Check if we already have this information cached
            if zone in self.gpu_cache:
                return self.gpu_cache[zone]
            request = compute_v1.ListAcceleratorTypesRequest(
                project=self.project_id,
                zone=zone
            )
            types = self.gpu_types.list(request=request)
            available_gpus = []

            for t in types:
                if t.name in GPU_TYPES:
                    available_gpus.append(t.name)

            # Cache the results
            self.gpu_cache[zone] = available_gpus
            duration_ms = (time.time() - start_time) * 1000
            self.performance["list_accelerator_types"].append(duration_ms)
            return available_gpus
        except GoogleAPIError as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance["list_accelerator_types"].append(duration_ms)
            logger.warning(f"Failed to list accelerator types in zone {zone}: {e}")
            return []

    def check_quota_region(self, region: str, gpu_type: str) -> float:
        """
        method 2: check quota
        """
        start_time = time.time()
        try:
            # Check if we have cached quota information for this region
            if region in self.gpu_quotas_cache:
                return self.gpu_quotas_cache.get(region, {}).get(gpu_type, 0.0)

            # Map GPU types to quota metric names
            gpu_to_quota = {
                "nvidia-tesla-t4": "NVIDIA_T4_GPUS",
                "nvidia-tesla-v100": "NVIDIA_V100_GPUS",
                "nvidia-tesla-p100": "NVIDIA_P100_GPUS",
                "nvidia-tesla-p4": "NVIDIA_P4_GPUS",
                "nvidia-tesla-a100": "NVIDIA_A100_GPUS",
                "nvidia-a100-80gb": "NVIDIA_A100_80GB_GPUS",
                "nvidia-l4": "NVIDIA_L4_GPUS"
            }

            request = compute_v1.GetRegionRequest(
                project=self.project_id,
                region=region
            )
            region_info = self.regions.get(request=request)

            # Cache all GPU quotas for this region at once
            if region not in self.gpu_quotas_cache:
                self.gpu_quotas_cache[region] = {}

            for quota in region_info.quotas:
                for gpu, quota_name in gpu_to_quota.items():
                    if quota.metric == quota_name:
                        available = quota.limit - quota.usage
                        self.gpu_quotas_cache[region][gpu] = available

            duration_ms = (time.time() - start_time) * 1000
            self.performance["check_quota"].append(duration_ms)
            return self.gpu_quotas_cache.get(region, {}).get(gpu_type, 0.0)
        except GoogleAPIError as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance["check_quota"].append(duration_ms)
            logger.warning(f"Failed to check GPU quota in region {region}: {e}")
            return 0.0

    def create_delete_vm(self, zone: str, gpu_type: str, gpu_count: int = 1) -> Tuple[bool, Optional[str], Optional[str]]:
        # create vm, immediately delete
        start_time = time.time()
        instance_name = f"gpu-test-{uuid.uuid4().hex[:8]}"

        try:
            vm_config = VM_CONFIGS.get(gpu_type, VM_CONFIGS["nvidia-tesla-t4"])
            machine_type = vm_config["machine_type"]

            # Create
            instance = compute_v1.Instance()
            instance.name = instance_name
            instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

            # all configs for vm
            disk = compute_v1.AttachedDisk()
            initialize_params = compute_v1.AttachedDiskInitializeParams()
            initialize_params.disk_size_gb = 100
            initialize_params.source_image = self.latest_img
            disk.initialize_params = initialize_params
            disk.auto_delete = True
            disk.boot = True
            instance.disks = [disk]
            network_interface = compute_v1.NetworkInterface()
            access_config = compute_v1.AccessConfig()
            network_interface.access_configs = [access_config]
            instance.network_interfaces = [network_interface]
            scheduling = compute_v1.Scheduling()
            scheduling.preemptible = True
            scheduling.automatic_restart = False
            instance.scheduling = scheduling
            guest_accelerator = compute_v1.AcceleratorConfig()
            guest_accelerator.accelerator_count = gpu_count
            guest_accelerator.accelerator_type = f"projects/{self.project_id}/zones/{zone}/acceleratorTypes/{gpu_type}"
            instance.guest_accelerators = [guest_accelerator]

            # create
            request = compute_v1.InsertInstanceRequest(project=self.project_id,zone=zone,instance_resource=instance,)
            operation = self.compute_client.insert(request=request, timeout=30)
            logger.info(f"success -  created VM {instance_name}, now deleting ")

            # delete
            delete_request = compute_v1.DeleteInstanceRequest(project=self.project_id,zone=zone,instance=instance.name)
            del_res = self.compute_client.delete(request=delete_request)
            logger.info(f"deleted {instance_name}")
            duration = (time.time() - start_time) * 1000
            self.performance["create_and_delete_vm"].append(duration)
            return True, None, None
        except PermissionDenied as e:
            error = "no permission"
            duration = (time.time() - start_time) * 1000
            self.performance["create_and_delete_vm"].append(duration)
            return False, error, str(e)
        except ResourceExhausted as e:
            error = "quota exceeded"
            duration = (time.time() - start_time) * 1000
            self.performance["create_and_delete_vm"].append(duration)
            return False, error, str(e)
        except InvalidArgument as e:
            error_msg = str(e)
            duration = (time.time() - start_time) * 1000
            self.performance["create_and_delete_vm"].append(duration)
            if "is not valid" in error_msg or "was not found" in error_msg:
                error = "GPU Not Available"
            else:
                error = "invalid"
            return False, error, error_msg
        except requests.exceptions.Timeout:
            return False, "Timeout", "API request timed out"
        except GoogleAPIError as e:
            error_msg = str(e)
            duration = (time.time() - start_time) * 1000
            self.performance["create_and_delete_vm"].append(duration)
            if "Quota" in error_msg:
                error = "quota exceeded"
            elif "not found" in error_msg:
                error = "gpu not available"
            elif "exceeds your limit" in error_msg:
                error = "resource limit exceeded"
            else:
                error = "api error"
            return False, error, error_msg

    def check_gpu_avail_zone(self, zone: str, gpu_type: str) -> Result:
        try:
            region = self.zone_to_region.get(zone)
            if not region:
                return Result(
                    zone=zone,
                    gpu_type=gpu_type,
                    is_available=False,
                    method="zone_mapping",
                    error_category="Invalid Zone",
                    error_message="Could not map zone to region"
                )
            # 1. check gpu availability
            start_time = time.time()
            available_gpus = self.check_gpu_zone(zone)
            method1_latency = (time.time() - start_time) * 1000
            method1_result = gpu_type in available_gpus

            # 2. check quota
            start_time = time.time()
            quota = self.check_quota_region(region, gpu_type)
            method2_latency = (time.time() - start_time) * 1000
            method2_result = quota > 0

            # if gpu or quota not available, don't create
            if not method1_result and not method2_result:
                return Result(
                    zone=zone,
                    gpu_type=gpu_type,
                    is_available=False,
                    method="accelerator_and_quota",
                    error_category="No GPUs Available",
                    error_message=f"GPU {gpu_type} not listed in available accelerators and no quota available",
                    latency=max(method1_latency, method2_latency)
                )

            # 3. dry-run (create and delete)
            start_time = time.time()
            is_available, error_category, error_message = self.create_delete_vm(
                zone=zone,
                gpu_type=gpu_type
            )
            latency3 = (time.time() - start_time) * 1000

            result = Result(
                zone=zone,
                gpu_type=gpu_type,
                is_available=is_available,
                method="create_and_delete_vm",
                error_category=error_category,
                error_message=error_message,
                latency=latency3
            )
            if is_available:
                self.successes.append(result)
            return result
        except Exception as e:
            logger.error(f"error gpu in {zone}: {e}")
            return Result(
                zone=zone,
                gpu_type=gpu_type,
                is_available=False,
                method="error",
                error_category="Exception",
                error_message=str(e)
            )

    def check_gpu_avail_parallel(self) -> Dict[str, Dict[str, List[Result]]]:
        results: Dict[str, Dict[str, List[Result]]] = {}

        priority_regions = ["us-central1", "us-east1", "us-west1", "europe-west4", "asia-east1"]
        priority_zones = []

        for zone in self.all_zones:
            region = self.zone_to_region.get(zone)
            if region in priority_regions:
                priority_zones.append(zone)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            for zone in priority_zones:
                for gpu_type in GPU_TYPES:
                    futures.append(executor.submit(self.check_gpu_avail_zone, zone, gpu_type))
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result.is_available:
                    if result.zone not in results:
                        results[result.zone] = {}
                    if result.gpu_type not in results[result.zone]:
                        results[result.zone][result.gpu_type] = []
                    results[result.zone][result.gpu_type].append(result)
                    return results
        remaining_zones = [z for z in self.all_zones if z not in priority_zones]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            for zone in remaining_zones:
                for gpu_type in GPU_TYPES:
                    futures.append(executor.submit(self.check_gpu_avail_zone, zone, gpu_type))
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result.is_available:
                    if result.zone not in results:
                        results[result.zone] = {}
                    if result.gpu_type not in results[result.zone]:
                        results[result.zone][result.gpu_type] = []
                    results[result.zone][result.gpu_type].append(result)
                    # returning early because of quota
                    return results
        logger.warning("No GPUs found")
        return results

    def create_vms_avail_gpus(self) -> None:
        """create VMs for available GPUs from successful checks"""
        if not self.successes:
            logger.warning("no available gpus")
            return

        gpu_types = list(set(result.gpu_type for result in self.successes))

        # at least 2 gpu types
        if len(gpu_types) < 2:
            logger.warning(f"found {len(gpu_types)} GPU types, want at least 2")

        vm_create_attempts = []
        for gpu_type in gpu_types:
            type_results = [r for r in self.successes if r.gpu_type == gpu_type]
            if type_results:
                vm_create_attempts.append(type_results[0])

    def compare_methods(self) -> None:
        methods = {
            "list_accelerator_types": "1. list gpu types",
            "check_quota": "2. check quota",
            "create_and_delete_vm": "3. create and delete VM"
        }

        print("\nMethod Comparison:")
        print(f"{'Method':<30} | {'Avg Time (ms)':<15} | {'Calls':<10}")
        print("-" * 60)

        for key, name in methods.items():
            times = self.performance.get(key, [])
            if times:
                avg_time = sum(times) / len(times)
                calls = len(times)
                print(f"{name:<30} | {avg_time:<15.2f} | {calls:<10}")
            else:
                print(f"{name:<30} | {'N/A':<15} | {'0':<10}")

        print("-" * 60)

        list_gpu_times = self.performance.get("list_accelerator_types", [])
        quota_times = self.performance.get("check_quota", [])
        vm_times = self.performance.get("create_and_delete_vm", [])

        if list_gpu_times and quota_times and vm_times:
            list_accel_avg = sum(list_gpu_times) / len(list_gpu_times)
            quota_avg = sum(quota_times) / len(quota_times)
            vm_avg = sum(vm_times) / len(vm_times)

    def check_first_zones(self, n: int = 10) -> List[Result]:
        results = []
        zones = self.get_all_zones()[:n]  # take the first n zones
        for zone in zones:
            for gpu_type in GPU_TYPES:
                result = self.check_gpu_avail_zone(zone, gpu_type)
                results.append(result)
        return results

def print_res(results: List[Result]):
    table = []
    for res in results:
        gpu_listed = "Yes" if res.method == "accelerator_and_quota" and res.error_category is None else "No"
        allocated = "Yes" if res.is_available else "No"
        reason = res.error_category if res.error_category else (res.error_message or "")
        table.append([res.zone, gpu_listed, allocated, reason, f"{res.latency:.2f} ms"])

    headers = ["Zone", "GPU available", "GPU allocated successfully", "Reason for failure", "Time taken"]
    print(tabulate(table, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description='gpu checkerr')
    parser.add_argument('--project', type=str, required=True, help='proj id')
    args = parser.parse_args()

    checker = Checker(project_id=args.project)
    if not checker.check_permissions():
        logger.error("permission check failed")
        return

    logger.info("starting check")
    results = checker.check_first_zones(n=10)
    results2 = checker.check_gpu_avail_parallel()
    print_res(results)
    checker.create_vms_avail_gpus()
    checker.compare_methods()


if __name__ == "__main__":
    main()