import logging
import os

from pydantic import BaseModel, ConfigDict

from microscopy_proc.constants import ProjFolders, RefFolders
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.io_utils import read_json, write_json


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

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
    points_check: str
    heatmap_check: str
    points_trfm_check: str
    heatmap_trfm_check: str

    @classmethod
    def get_proj_fp_model(cls, proj_dir):
        return cls(
            # CONFIGS
            config_params=os.path.join(proj_dir, "config_params.json"),
            # MY ATLAS AND ELASTIX PARAMS FILES
            ref=os.path.join(proj_dir, "registration", "0a_reference.tif"),
            annot=os.path.join(proj_dir, "registration", "0b_annotation.tif"),
            map=os.path.join(proj_dir, "registration", "0c_mapping.json"),
            affine=os.path.join(proj_dir, "registration", "0d_align_affine.txt"),
            bspline=os.path.join(proj_dir, "registration", "0e_align_bspline.txt"),
            # RAW IMG FILE
            raw=os.path.join(proj_dir, "raw.zarr"),
            # REGISTRATION PROCESSING FILES
            downsmpl1=os.path.join(proj_dir, "registration", "1_downsmpl1.tif"),
            downsmpl2=os.path.join(proj_dir, "registration", "2_downsmpl2.tif"),
            trimmed=os.path.join(proj_dir, "registration", "3_trimmed.tif"),
            regresult=os.path.join(proj_dir, "registration", "4_regresult.tif"),
            # WHOLE MASK
            premask_blur=os.path.join(proj_dir, "mask", "1_premask_blur.tif"),
            mask=os.path.join(proj_dir, "mask", "2_mask_trimmed.tif"),
            outline=os.path.join(proj_dir, "mask", "3_outline_reg.tif"),
            mask_reg=os.path.join(proj_dir, "mask", "4_mask_reg.tif"),
            mask_df=os.path.join(proj_dir, "mask", "5_mask.parquet"),
            # CELL COUNTING ARRAY FILES
            overlap=os.path.join(proj_dir, "cellcount", "0_overlap.zarr"),
            bgrm=os.path.join(proj_dir, "cellcount", "1_bgrm.zarr"),
            dog=os.path.join(proj_dir, "cellcount", "2_dog.zarr"),
            adaptv=os.path.join(proj_dir, "cellcount", "3_adaptv.zarr"),
            threshd=os.path.join(proj_dir, "cellcount", "4_threshd.zarr"),
            threshd_volumes=os.path.join(
                proj_dir, "cellcount", "5_threshd_volumes.zarr"
            ),
            threshd_filt=os.path.join(proj_dir, "cellcount", "6_threshd_filt.zarr"),
            maxima=os.path.join(proj_dir, "cellcount", "7_maxima.zarr"),
            wshed_volumes=os.path.join(proj_dir, "cellcount", "8_wshed_volumes.zarr"),
            wshed_filt=os.path.join(proj_dir, "cellcount", "9_wshed_filt.zarr"),
            threshd_final=os.path.join(proj_dir, "cellcount", "10_threshd_f.zarr"),
            maxima_final=os.path.join(proj_dir, "cellcount", "10_maxima_f.zarr"),
            wshed_final=os.path.join(proj_dir, "cellcount", "10_wshed_f.zarr"),
            # CELL COUNTING DF FILES
            maxima_df=os.path.join(proj_dir, "analysis", "11_maxima.parquet"),
            cells_raw_df=os.path.join(proj_dir, "analysis", "11_cells_raw.parquet"),
            cells_trfm_df=os.path.join(proj_dir, "analysis", "12_cells_trfm.parquet"),
            cells_df=os.path.join(proj_dir, "analysis", "13_cells.parquet"),
            cells_agg_df=os.path.join(proj_dir, "analysis", "14_cells_agg.parquet"),
            cells_agg_csv=os.path.join(proj_dir, "analysis", "15_cells_agg.csv"),
            # VISUAL CHECK FROM CELL DF FILES
            points_check=os.path.join(proj_dir, "visual_check", "points.zarr"),
            heatmap_check=os.path.join(proj_dir, "visual_check", "heatmap.zarr"),
            points_trfm_check=os.path.join(
                proj_dir, "visual_check", "points_trfm.zarr"
            ),
            heatmap_trfm_check=os.path.join(
                proj_dir, "visual_check", "heatmap_trfm.zarr"
            ),
        )


def get_proj_fp_model(proj_dir: str):
    return ProjFpModel.get_proj_fp_model(proj_dir)


def make_proj_dirs(proj_dir):
    for folder in ProjFolders:
        os.makedirs(os.path.join(proj_dir, folder.value), exist_ok=True)


def update_configs(pfm: ProjFpModel, **kwargs) -> ConfigParamsModel:
    """
    If config_params file does not exist, make a new one.

    Then updates the config_params file with the kwargs.
    If there are no kwargs, will not update the file
    (other than making it if it did not exist).
    """
    # Making registration params json
    try:  # If file exists
        rp = ConfigParamsModel.model_validate(read_json(pfm.config_params))
    except FileNotFoundError as e:  # If file does not exist
        logging.info(e)
        logging.info("Making new params json")
        rp = ConfigParamsModel()
        write_json(pfm.config_params, rp.model_dump())
    # Updating and saving configs if kwargs is not empty
    if kwargs != {}:
        rp = rp.model_validate(rp.model_copy(update=kwargs))
        write_json(pfm.config_params, rp.model_dump())
    # and returning the configs
    return rp
