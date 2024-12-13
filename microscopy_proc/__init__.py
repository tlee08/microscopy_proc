import importlib.util


# Checking if CPU or GPU version
def is_installed(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None


# Checking if gpu extra dependency is installed
GPU_ENABLED = is_installed("cupy") and is_installed("dask_cuda")
# Checking if elastix extra dependency is installed
ELASTIX_ENABLED = is_installed("SimpleITK")
