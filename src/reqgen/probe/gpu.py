"""GPU detection via NVIDIA Management Library (NVML), through the pynvml bindings."""

from __future__ import annotations

from reqgen.probe.schema import GPUInfo


def detect_gpus() -> tuple[list[GPUInfo], list[str]]:
    """
    Query all NVIDIA GPUs visible on this system via NVML.

    Returns:
        (gpus, warnings) — a list of GPUInfo objects (empty if none found or
        NVML unavailable), and a list of non-fatal warning strings explaining
        anything that went wrong along the way.
    """
    warnings: list[str] = []
    gpus: list[GPUInfo] = []

    try:
        import pynvml
    except ImportError:
        warnings.append(
            "pynvml is not installed. Run `uv pip install pynvml` to enable GPU detection."
        )
        return gpus, warnings

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        warnings.append(f"NVML failed to initialize (no NVIDIA driver/GPU found?): {e}")
        return gpus, warnings

    try:
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode()

        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            compute_capability = None
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"
            except pynvml.NVMLError:
                pass  # not fatal — some driver/lib combos don't expose this

            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    driver_version=driver_version,
                    vram_total_mb=mem_info.total // (1024 * 1024),
                    vram_free_mb=mem_info.free // (1024 * 1024),
                    compute_capability=compute_capability,
                )
            )

    except pynvml.NVMLError as e:
        warnings.append(f"Error while querying GPU details: {e}")

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass  # already shut down or never fully initialized — safe to ignore

    if not gpus and not warnings:
        warnings.append("No NVIDIA GPUs detected on this system.")

    return gpus, warnings