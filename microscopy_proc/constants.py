import os
import pathlib

PROC_CHUNKS = (500, 1000, 1000)
# PROC_CHUNKS = (500, 1200, 1200)


# DEPTH = 10
DEPTH = 50


ROWSPPART = 10000000


CELL_MEASURES = {
    "z": "count",
    "size": "volume",
    "sum_itns": "sum",
    # "max_itns": "max",
}


CONFIGS_DIR = ".microscopy_proc"


# Download Allen Brain Atlas atlas resources
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")

TEMP_DIR = os.path.join(pathlib.Path.home(), ".microscopy_proc_temp")
