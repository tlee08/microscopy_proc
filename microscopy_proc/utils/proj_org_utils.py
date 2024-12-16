import os
from enum import Enum

from pydantic import BaseModel, ConfigDict

from microscopy_proc.utils.logging_utils import init_logger

# TODO: add 10_adaptv_f.zarr and move the xxx_f.zarr files to a new folder (e.g. "cellcount_final")
# NOTE: this allows "cellcount" to be removed to save space when pipeline is completed and output checked

logger = init_logger()


class ProjSubdirs(Enum):
    REGISTRATION = "registration"
    MASK = "mask"
    CELLCOUNT = "cellcount"
    CELLCOUNT_TUNING = "cellcount_tuning"
    ANALYSIS = "analysis"
    VISUALISATION = "visualisation"
    COMBINED = "combined"


class RefFolders(Enum):
    REFERENCE = "reference"
    ANNOTATION = "annotation"
    MAPPING = "region_mapping"
    ELASTIX = "elastix_params"


class RefFpModel(BaseModel):
    """
    Pydantic model for reference file paths.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # ATLAS FILES ORIGIN
    ref: str
    annot: str
    map: str
    # ELASTIX PARAM FILES ORIGIN
    affine: str
    bspline: str

    @classmethod
    def get_ref_fp_model(cls, atlas_dir, ref_v, annot_v, map_v):
        """
        atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"

        Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
        """
        return cls(
            ref=os.path.join(atlas_dir, RefFolders.REFERENCE.value, f"{ref_v}.tif"),
            annot=os.path.join(
                atlas_dir, RefFolders.ANNOTATION.value, f"{annot_v}.tif"
            ),
            map=os.path.join(atlas_dir, RefFolders.MAPPING.value, f"{map_v}.json"),
            affine=os.path.join(
                atlas_dir, RefFolders.ELASTIX.value, "align_affine.txt"
            ),
            bspline=os.path.join(
                atlas_dir, RefFolders.ELASTIX.value, "align_bspline.txt"
            ),
        )


class ProjFpModel(BaseModel):
    """
    Pydantic model for project file paths.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_default=True,
        use_enum_values=True,
    )

    # ROOT DIR
    root_dir: str
    # CONFIGS
    config_params: str
    # ATLAS AND ELASTIX PARAMS FILES
    ref: str
    annot: str
    map: str
    affine: str
    bspline: str
    # RAW IMG FILE
    raw: str
    # REGISTRATION PROCESSING FILES
    downsmpl1: str
    downsmpl2: str
    trimmed: str
    regresult: str
    # WHOLE MASK
    premask_blur: str
    mask: str
    outline: str
    mask_reg: str
    mask_df: str
    # CELL COUNTING ARRAY FILES
    overlap: str
    bgrm: str
    dog: str
    adaptv: str
    threshd: str
    threshd_volumes: str
    threshd_filt: str
    maxima: str
    wshed_volumes: str
    wshed_filt: str
    threshd_final: str
    maxima_final: str
    wshed_final: str
    # CELL COUNTING DF FILES
    maxima_df: str
    cells_raw_df: str
    cells_trfm_df: str
    cells_df: str
    cells_agg_df: str
    cells_agg_csv: str
    # VISUAL CHECK FROM CELL DF FILES
    points_raw: str
    heatmap_raw: str
    points_trfm: str
    heatmap_trfm: str
    # COMBINED ARRAYS
    combined_reg: str
    combined_cellc: str
    combined_points: str

    @classmethod
    def get_proj_fp_model(cls, proj_dir):
        reg_dir = ProjSubdirs.REGISTRATION.value
        mask_dir = ProjSubdirs.MASK.value
        cellc_dir = ProjSubdirs.CELLCOUNT.value
        analysis_dir = ProjSubdirs.ANALYSIS.value
        visual_dir = ProjSubdirs.VISUALISATION.value
        combined_dir = ProjSubdirs.COMBINED.value

        return cls(
            # ROOT DIR
            root_dir=proj_dir,
            # CONFIGS
            config_params=os.path.join(proj_dir, "config_params.json"),
            # MY ATLAS AND ELASTIX PARAMS FILES
            ref=os.path.join(proj_dir, reg_dir, "0a_reference.tif"),
            annot=os.path.join(proj_dir, reg_dir, "0b_annotation.tif"),
            map=os.path.join(proj_dir, reg_dir, "0c_mapping.json"),
            affine=os.path.join(proj_dir, reg_dir, "0d_align_affine.txt"),
            bspline=os.path.join(proj_dir, reg_dir, "0e_align_bspline.txt"),
            # RAW IMG FILE
            raw=os.path.join(proj_dir, "raw.zarr"),
            # REGISTRATION PROCESSING FILES
            downsmpl1=os.path.join(proj_dir, reg_dir, "1_downsmpl1.tif"),
            downsmpl2=os.path.join(proj_dir, reg_dir, "2_downsmpl2.tif"),
            trimmed=os.path.join(proj_dir, reg_dir, "3_trimmed.tif"),
            regresult=os.path.join(proj_dir, reg_dir, "4_regresult.tif"),
            # WHOLE MASK
            premask_blur=os.path.join(proj_dir, mask_dir, "1_premask_blur.tif"),
            mask=os.path.join(proj_dir, mask_dir, "2_mask_trimmed.tif"),
            outline=os.path.join(proj_dir, mask_dir, "3_outline_reg.tif"),
            mask_reg=os.path.join(proj_dir, mask_dir, "4_mask_reg.tif"),
            mask_df=os.path.join(proj_dir, mask_dir, "5_mask.parquet"),
            # CELL COUNTING ARRAY FILES
            overlap=os.path.join(proj_dir, cellc_dir, "0_overlap.zarr"),
            bgrm=os.path.join(proj_dir, cellc_dir, "1_bgrm.zarr"),
            dog=os.path.join(proj_dir, cellc_dir, "2_dog.zarr"),
            adaptv=os.path.join(proj_dir, cellc_dir, "3_adaptv.zarr"),
            threshd=os.path.join(proj_dir, cellc_dir, "4_threshd.zarr"),
            threshd_volumes=os.path.join(proj_dir, cellc_dir, "5_threshd_volumes.zarr"),
            threshd_filt=os.path.join(proj_dir, cellc_dir, "6_threshd_filt.zarr"),
            maxima=os.path.join(proj_dir, cellc_dir, "7_maxima.zarr"),
            wshed_volumes=os.path.join(proj_dir, cellc_dir, "8_wshed_volumes.zarr"),
            wshed_filt=os.path.join(proj_dir, cellc_dir, "9_wshed_filt.zarr"),
            # CELL COUNTING TRIMMED TO RAW ARRAY FILES
            threshd_final=os.path.join(proj_dir, cellc_dir, "10_threshd_f.zarr"),
            maxima_final=os.path.join(proj_dir, cellc_dir, "10_maxima_f.zarr"),
            wshed_final=os.path.join(proj_dir, cellc_dir, "10_wshed_f.zarr"),
            # CELL COUNTING DF FILES
            maxima_df=os.path.join(proj_dir, analysis_dir, "11_maxima.parquet"),
            cells_raw_df=os.path.join(proj_dir, analysis_dir, "11_cells_raw.parquet"),
            cells_trfm_df=os.path.join(proj_dir, analysis_dir, "12_cells_trfm.parquet"),
            cells_df=os.path.join(proj_dir, analysis_dir, "13_cells.parquet"),
            cells_agg_df=os.path.join(proj_dir, analysis_dir, "14_cells_agg.parquet"),
            cells_agg_csv=os.path.join(proj_dir, analysis_dir, "15_cells_agg.csv"),
            # VISUAL CHECK
            points_raw=os.path.join(proj_dir, visual_dir, "points_raw.zarr"),
            heatmap_raw=os.path.join(proj_dir, visual_dir, "heatmap_raw.zarr"),
            points_trfm=os.path.join(proj_dir, visual_dir, "points_trfm.tif"),
            heatmap_trfm=os.path.join(proj_dir, visual_dir, "heatmap_trfm.tif"),
            # COMBINE ARRAYS
            combined_reg=os.path.join(proj_dir, combined_dir, "combined_reg.tif"),
            combined_cellc=os.path.join(proj_dir, combined_dir, "combined_cellc.tif"),
            combined_points=os.path.join(proj_dir, combined_dir, "combined_points.tif"),
        )

    def copy(self):
        return self.model_validate(self.model_dump())

    def _convert_to(self, cellc_dir: str):
        # Getting root_dir
        root_dir = self.root_dir
        # Converting raw filepath
        # TODO: a better way to do this with encapsulation and not hardcoding
        if cellc_dir == ProjSubdirs.CELLCOUNT.value:
            self.raw = os.path.join(root_dir, "raw.zarr")
        elif cellc_dir == ProjSubdirs.CELLCOUNT_TUNING.value:
            self.raw = os.path.join(root_dir, "raw_tuning.zarr")
        # Converting all the cellcount filepaths to the new directory
        self.overlap = os.path.join(root_dir, cellc_dir, "0_overlap.zarr")
        self.bgrm = os.path.join(root_dir, cellc_dir, "1_bgrm.zarr")
        self.dog = os.path.join(root_dir, cellc_dir, "2_dog.zarr")
        self.adaptv = os.path.join(root_dir, cellc_dir, "3_adaptv.zarr")
        self.threshd = os.path.join(root_dir, cellc_dir, "4_threshd.zarr")
        self.threshd_volumes = os.path.join(
            root_dir, cellc_dir, "5_threshd_volumes.zarr"
        )
        self.threshd_filt = os.path.join(root_dir, cellc_dir, "6_threshd_filt.zarr")
        self.maxima = os.path.join(root_dir, cellc_dir, "7_maxima.zarr")
        self.wshed_volumes = os.path.join(root_dir, cellc_dir, "8_wshed_volumes.zarr")
        self.wshed_filt = os.path.join(root_dir, cellc_dir, "9_wshed_filt.zarr")
        # Converting all the cellcount trimmed to raw filepaths to the new directory
        self.threshd_final = os.path.join(root_dir, cellc_dir, "10_threshd_f.zarr")
        self.maxima_final = os.path.join(root_dir, cellc_dir, "10_maxima_f.zarr")
        self.wshed_final = os.path.join(root_dir, cellc_dir, "10_wshed_f.zarr")
        # Returning
        return self

    def convert_to_tuning(self):
        self._convert_to(ProjSubdirs.CELLCOUNT_TUNING.value)
        return self

    def convert_to_processing(self):
        self._convert_to(ProjSubdirs.CELLCOUNT.value)
        return self


def make_proj_dirs(pfm: ProjFpModel):
    for folder in ProjSubdirs:
        os.makedirs(os.path.join(pfm.root_dir, folder.value), exist_ok=True)
