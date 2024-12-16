import os
from enum import Enum

from microscopy_proc.utils.logging_utils import init_logger

# TODO: add 10_adaptv_f.zarr and move the xxx_f.zarr files to a new folder (e.g. "cellcount_final")
# NOTE: this allows "cellcount" to be removed to save space when pipeline is completed and output checked

logger = init_logger()


class ProjSubdirs(Enum):
    RAW = "raw"
    REGISTRATION = "registration"
    MASK = "mask"
    CELLCOUNT = "cellcount"
    ANALYSIS = "analysis"
    VISUALISATION = "visualisation"
    COMBINED = "combined"


class ProjSubdirsTuning(Enum):
    RAW = "raw_tuning"
    REGISTRATION = "registration"
    MASK = "mask"
    CELLCOUNT = "cellcount_tuning"
    ANALYSIS = "analysis_tuning"
    VISUALISATION = "visualisation_tuning"
    COMBINED = "combined_tuning"


class RefFolders(Enum):
    REFERENCE = "reference"
    ANNOTATION = "annotation"
    MAPPING = "region_mapping"
    ELASTIX = "elastix_params"


class RefFpModel:
    """
    Model for reference file paths.
    """

    def __init__(self, atlas_dir, ref_v, annot_v, map_v):
        """
        atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"

        Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
        """
        # Quick variables for folders
        reference_dir = RefFolders.REFERENCE.value
        annotation_dir = RefFolders.ANNOTATION.value
        mapping_dir = RefFolders.MAPPING.value
        elastix_dir = RefFolders.ELASTIX.value
        # Setting file paths
        # ATLAS FILES ORIGIN
        self.ref = os.path.join(atlas_dir, reference_dir, f"{ref_v}.tif")
        self.annot = os.path.join(atlas_dir, annotation_dir, f"{annot_v}.tif")
        self.map = os.path.join(atlas_dir, mapping_dir, f"{map_v}.json")
        # ELASTIX PARAM FILES ORIGIN
        self.affine = os.path.join(atlas_dir, elastix_dir, "align_affine.txt")
        self.bspline = os.path.join(atlas_dir, elastix_dir, "align_bspline.txt")


class ProjFpModelBase:
    """
    Model for project file paths.
    """

    # ROOT DIR
    root_dir: str
    # FROM root_dir AND subdirs
    # CONFIGS
    config_params: str
    # RAW IMG FILE
    raw: str | None
    # MY ATLAS AND ELASTIX PARAMS FILES AND REGISTRATION PROCESSING FILES
    ref: str | None
    annot: str | None
    map: str | None
    affine: str | None
    bspline: str | None
    downsmpl1: str | None
    downsmpl2: str | None
    trimmed: str | None
    regresult: str | None
    # WHOLE MASK
    premask_blur: str | None
    mask: str | None
    outline: str | None
    mask_reg: str | None
    mask_df: str | None
    # CELL COUNTING ARRAY FILES AND CELL COUNTING TRIMMED TO RAW ARRAY FILES
    overlap: str | None
    bgrm: str | None
    dog: str | None
    adaptv: str | None
    threshd: str | None
    threshd_volumes: str | None
    threshd_filt: str | None
    maxima: str | None
    wshed_volumes: str | None
    wshed_filt: str | None
    threshd_final: str | None
    maxima_final: str | None
    wshed_final: str | None
    # CELL COUNTING DF FILES
    maxima_df: str | None
    cells_raw_df: str | None
    cells_trfm_df: str | None
    cells_df: str | None
    cells_agg_df: str | None
    cells_agg_csv: str | None
    # VISUAL CHECK
    points_raw: str | None
    heatmap_raw: str | None
    points_trfm: str | None
    heatmap_trfm: str | None
    # COMBINE ARRAYS
    combined_reg: str | None
    combined_cellc: str | None
    combined_points: str | None

    def __init__(self, root_dir: str, subdirs):
        # ROOT DIR
        self.root_dir = root_dir
        # SUBDIRS
        self.subdirs = subdirs
        # FROM root_dir AND subdirs
        # CONFIGS
        self.config_params = os.path.join(root_dir, "config_params.json")
        # RAW IMG FILE
        raw = subdirs.RAW.value
        self.raw = os.path.join(root_dir, raw, "raw.zarr")
        # MY ATLAS AND ELASTIX PARAMS FILES AND REGISTRATION PROCESSING FILES
        reg = subdirs.REGISTRATION.value
        self.ref = os.path.join(root_dir, reg, "0a_reference.tif")
        self.annot = os.path.join(root_dir, reg, "0b_annotation.tif")
        self.map = os.path.join(root_dir, reg, "0c_mapping.json")
        self.affine = os.path.join(root_dir, reg, "0d_align_affine.txt")
        self.bspline = os.path.join(root_dir, reg, "0e_align_bspline.txt")
        self.downsmpl1 = os.path.join(root_dir, reg, "1_downsmpl1.tif")
        self.downsmpl2 = os.path.join(root_dir, reg, "2_downsmpl2.tif")
        self.trimmed = os.path.join(root_dir, reg, "3_trimmed.tif")
        self.regresult = os.path.join(root_dir, reg, "4_regresult.tif")
        # WHOLE MASK
        mask = subdirs.MASK.value
        self.premask_blur = os.path.join(root_dir, mask, "1_premask_blur.tif")
        self.mask = os.path.join(root_dir, mask, "2_mask_trimmed.tif")
        self.outline = os.path.join(root_dir, mask, "3_outline_reg.tif")
        self.mask_reg = os.path.join(root_dir, mask, "4_mask_reg.tif")
        self.mask_df = os.path.join(root_dir, mask, "5_mask.parquet")
        # CELL COUNTING ARRAY FILES AND CELL COUNTING TRIMMED TO RAW ARRAY FILES
        cellc = subdirs.CELLCOUNT.value
        self.overlap = os.path.join(root_dir, cellc, "0_overlap.zarr")
        self.bgrm = os.path.join(root_dir, cellc, "1_bgrm.zarr")
        self.dog = os.path.join(root_dir, cellc, "2_dog.zarr")
        self.adaptv = os.path.join(root_dir, cellc, "3_adaptv.zarr")
        self.threshd = os.path.join(root_dir, cellc, "4_threshd.zarr")
        self.threshd_volumes = os.path.join(root_dir, cellc, "5_threshd_volumes.zarr")
        self.threshd_filt = os.path.join(root_dir, cellc, "6_threshd_filt.zarr")
        self.maxima = os.path.join(root_dir, cellc, "7_maxima.zarr")
        self.wshed_volumes = os.path.join(root_dir, cellc, "8_wshed_volumes.zarr")
        self.wshed_filt = os.path.join(root_dir, cellc, "9_wshed_filt.zarr")
        self.threshd_final = os.path.join(root_dir, cellc, "10_threshd_f.zarr")
        self.maxima_final = os.path.join(root_dir, cellc, "10_maxima_f.zarr")
        self.wshed_final = os.path.join(root_dir, cellc, "10_wshed_f.zarr")
        # CELL COUNTING DF FILES
        analysis = subdirs.ANALYSIS.value
        self.maxima_df = os.path.join(root_dir, analysis, "1_maxima.parquet")
        self.cells_raw_df = os.path.join(root_dir, analysis, "1_cells_raw.parquet")
        self.cells_trfm_df = os.path.join(root_dir, analysis, "2_cells_trfm.parquet")
        self.cells_df = os.path.join(root_dir, analysis, "3_cells.parquet")
        self.cells_agg_df = os.path.join(root_dir, analysis, "4_cells_agg.parquet")
        self.cells_agg_csv = os.path.join(root_dir, analysis, "5_cells_agg.csv")
        # VISUAL CHECK
        visual = subdirs.VISUALISATION.value
        self.points_raw = os.path.join(root_dir, visual, "points_raw.zarr")
        self.heatmap_raw = os.path.join(root_dir, visual, "heatmap_raw.zarr")
        self.points_trfm = os.path.join(root_dir, visual, "points_trfm.tif")
        self.heatmap_trfm = os.path.join(root_dir, visual, "heatmap_trfm.tif")
        # COMBINE ARRAYS
        combined = subdirs.COMBINED.value
        self.combined_reg = os.path.join(root_dir, combined, "combined_reg.tif")
        self.combined_cellc = os.path.join(root_dir, combined, "combined_cellc.tif")
        self.combined_points = os.path.join(root_dir, combined, "combined_points.tif")

    def copy(self):
        return self.__init__(self.root_dir, self.subdirs)

    def make_subdirs(self):
        """
        Make project directories.
        """
        for folder in self.subdirs:
            os.makedirs(os.path.join(self.root_dir, folder.value), exist_ok=True)


class ProjFpModel(ProjFpModelBase):
    def __init__(self, root_dir: str):
        super().__init__(root_dir, ProjSubdirs)


class ProjFpModelTuning(ProjFpModelBase):
    def __init__(self, root_dir: str):
        super().__init__(root_dir, ProjSubdirsTuning)
        # NOTE: some of the files are not used in tuning mode. They are set to None
        # MY ATLAS AND ELASTIX PARAMS FILES AND REGISTRATION PROCESSING FILES
        self.ref = None
        self.annot = None
        self.map = None
        self.affine = None
        self.bspline = None
        self.downsmpl1 = None
        self.downsmpl2 = None
        self.trimmed = None
        self.regresult = None
        # WHOLE MASK
        self.premask_blur = None
        self.mask = None
        self.outline = None
        self.mask_reg = None
        self.mask_df = None
        # CELL COUNTING DF FILES
        self.cells_trfm_df = None
        self.cells_df = None
        self.cells_agg_df = None
        self.cells_agg_csv = None
        # VISUAL CHECK
        self.points_raw = None
        self.heatmap_raw = None
        self.points_trfm = None
        self.heatmap_trfm = None
        # COMBINE ARRAYS
        self.combined_reg = None
        self.combined_cellc = None
        self.combined_points = None
