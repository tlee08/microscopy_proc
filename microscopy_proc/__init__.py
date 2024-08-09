import importlib.util


# Checking if CPU or GPU version
def is_installed(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None


if is_installed("cupy"):
    INSTALLATION_TYPE = "gpu"
else:
    INSTALLATION_TYPE = "cpu"
print(f"Installation type: {INSTALLATION_TYPE}")
