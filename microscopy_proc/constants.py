import os
import pathlib
from enum import Enum

PROC_CHUNKS = (500, 1000, 1000)
# PROC_CHUNKS = (500, 1200, 1200)


# DEPTH = 10
DEPTH = 50


ROWSPPART = 10000000


class CellMeasures(Enum):
    z = "z"
    volume = "volume"
    sum_intensity = "sum_intensity"
    # max_intensity = "max_intensity"
    iov = "iov"


CELL_AGG_MAPPING = {
    CellMeasures.z.value: "count",
    CellMeasures.volume.value: "sum",
    CellMeasures.sum_intensity.value: "sum",
    # CellMeasures.max_intensity.value: "max",
}


class ProjFolders(Enum):
    REGISTRATION = "registration"
    MASK = "mask"
    CELLCOUNT = "cellcount"
    ANALYSIS = "analysis"
    VISUALISATION = "visualisation"


class RefFolders(Enum):
    REFERENCE = "reference"
    ANNOTATION = "annotation"
    MAPPING = "region_mapping"
    ELASTIX = "elastix_params"


CONFIGS_DIR = ".microscopy_proc"


# Download Allen Brain Atlas atlas resources
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")

TEMP_DIR = os.path.join(pathlib.Path.home(), ".microscopy_proc_temp")
