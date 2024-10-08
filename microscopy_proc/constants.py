import os
import pathlib
from enum import Enum

import numpy as np

PROC_CHUNKS = (500, 1000, 1000)
# PROC_CHUNKS = (500, 1200, 1200)

# DEPTH = 10
DEPTH = 50

ROWS_PARTITION = 10000000


class RefFolders(Enum):
    REFERENCE = "reference"
    ANNOTATION = "annotation"
    MAPPING = "region_mapping"
    ELASTIX = "elastix_params"


class RefVersions(Enum):
    AVERAGE_TEMPLATE_25 = "average_template_25"
    ARA_NISSL_25 = "ara_nissl_25"


class AnnotVersions(Enum):
    CCF_2017_25 = "ccf_2017_25"
    CCF_2016_25 = "ccf_2016_25"
    CCF_2015_25 = "ccf_2015_25"


class MapVersions(Enum):
    ABA_ANNOTATIONS = "ABA_annotations"
    CM_ANNOTATIONS = "CM_annotations"


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


class AnnotExtraColumns(Enum):
    PARENT_ID = "parent_id"
    PARENT_ACRONYM = "parent_acronym"
    CHILDREN = "children"


ANNOT_COLUMNS_TYPES = {
    AnnotColumns.ID.value: np.float64,
    AnnotColumns.ATLAS_ID.value: np.float64,
    AnnotColumns.ONTOLOGY_ID.value: np.float64,
    AnnotColumns.ACRONYM.value: str,
    AnnotColumns.NAME.value: str,
    AnnotColumns.COLOR_HEX_TRIPLET.value: str,
    AnnotColumns.GRAPH_ORDER.value: np.float64,
    AnnotColumns.ST_LEVEL.value: np.float64,
    AnnotColumns.HEMISPHERE_ID.value: np.float64,
    AnnotColumns.PARENT_STRUCTURE_ID.value: np.float64,
}

ANNOT_COLUMNS_FINAL = [
    AnnotColumns.NAME.value,
    AnnotColumns.ACRONYM.value,
    AnnotColumns.COLOR_HEX_TRIPLET.value,
    AnnotColumns.PARENT_STRUCTURE_ID.value,
    AnnotExtraColumns.PARENT_ACRONYM.value,
]


class CellColumns(Enum):
    COUNT = "count"
    VOLUME = "volume"
    SUM_INTENSITY = "sum_intensity"
    # MAX_INTENSITY = "max_intensity"
    IOV = "iov"


CELL_AGG_MAPPINGS = {
    CellColumns.COUNT.value: "sum",
    CellColumns.VOLUME.value: "sum",
    CellColumns.SUM_INTENSITY.value: "sum",
    # CellMeasures.MAX_INTENSITY.value: "max",
}

MASK_VOLUME = "volume"


class MaskColumns(Enum):
    VOLUME_ANNOT = f"{MASK_VOLUME}_annot"
    VOLUME_MASK = f"{MASK_VOLUME}_mask"
    VOLUME_PROP = f"{MASK_VOLUME}_prop"


class ProjFolders(Enum):
    REGISTRATION = "registration"
    MASK = "mask"
    CELLCOUNT = "cellcount"
    ANALYSIS = "analysis"
    VISUALISATION = "visualisation"


CONFIGS_DIR = ".microscopy_proc"


# Download Allen Brain Atlas atlas resources
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")

TEMP_DIR = os.path.join(pathlib.Path.home(), ".microscopy_proc_temp")
