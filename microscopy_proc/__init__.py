import importlib.util
import os

from microscopy_proc.constants import CONFIGS_DIR

# First-time install code
# Download Allen Brain Atlas atlas resources
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
# home_dir = os.path.expanduser("~")
# if not os.path.exists(os.path.join(home_dir, CONFIGS_DIR)):
#     os.makedirs(os.path.join(home_dir, CONFIGS_DIR), exist_ok=True)
# if not os.path.exists(os.path.join(home_dir, CONFIGS_DIR, "atlas_resources")):
resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")


# Checking if CPU or GPU version
def is_installed(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None


if is_installed("cupy"):
    INSTALLATION_TYPE = "gpu"
else:
    INSTALLATION_TYPE = "cpu"
print(f"Installation type: {INSTALLATION_TYPE}")
