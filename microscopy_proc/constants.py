import os
import pathlib
from enum import Enum

PROC_CHUNKS = (500, 1000, 1000)
# PROC_CHUNKS = (500, 1200, 1200)

# DEPTH = 10
DEPTH = 50

ROWSPPARTITION = 10000000


class Coords(Enum):
    X = "x"
    Y = "y"
    Z = "z"


TRFM = "trfm"

CELL_IDX_NAME = "label"


class AnnotColumns(Enum):
    ID = "id"
    ATLAS_ID = "atlas_id"
    ONTOLOGY_ID = "ontology_id"
    ACRONYM = "acronym"
    NAME = "name"
    COLOR_HEX_TRIPLET = "color_hex_triplet"
    GRAPH_ORDER = "graph_order"
    ST_LEVEL = "st_level"
    HEMISPHERE_ID = "hemisphere_id"
    PARENT_STRUCTURE_ID = "parent_structure_id"


class CellMeasures(Enum):
    Z = "z"
    VOLUME = "volume"
    SUM_INTENSITY = "sum_intensity"
    # MAX_INTENSITY = "max_intensity"
    IOV = "iov"


CELL_AGG_MAPPING = {
    CellMeasures.Z.value: "count",
    CellMeasures.VOLUME.value: "sum",
    CellMeasures.SUM_INTENSITY.value: "sum",
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
