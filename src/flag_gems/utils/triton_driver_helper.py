try:
    from flag_gems.runtime import torch_device_fn

    get_device_properties = torch_device_fn.get_device_properties
except AttributeError:
    try:
        import triton

        get_device_properties = triton.runtime.driver.active.utils.get_device_properties
    except RuntimeError:
        # CPU/spacemit backend: no GPU driver; return a stub so imports succeed.
        def get_device_properties(device):
            return type(
                "CpuDeviceProperties",
                (),
                {
                    "max_shared_mem": 0,
                    "multiprocessor_count": 1,
                    "multi_processor_count": 1,
                    "total_memory": 64 * 1024**3,
                },
            )()
