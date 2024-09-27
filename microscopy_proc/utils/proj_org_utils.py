import logging
import os

from microscopy_proc.constants import RESOURCES_DIR
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.io_utils import read_json, write_json


def get_ref_fp_dict(atlas_dir=None, ref_v=None, annot_v=None, map_v=None):
    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    # Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

    # NAMES OF ATLAS FILES USED
    # resources dir
    atlas_dir = RESOURCES_DIR if atlas_dir is None else atlas_dir
    # "average_template_25", "ara_nissl_25"
    ref_v = "average_template_25" if ref_v is None else ref_v
    # "ccf_2017_25", "ccf_2016_25", "ccf_2015_25"
    annot_v = "ccf_2016_25" if annot_v is None else ref_v
    # "ABA_annotations", "CM_annotations"
    map_v = "ABA_annotations" if map_v is None else ref_v

    return {
        # ATLAS FILES ORIGIN
        "ref": os.path.join(atlas_dir, "reference", f"{ref_v}.tif"),
        "annot": os.path.join(atlas_dir, "annotation", f"{annot_v}.tif"),
        "map": os.path.join(atlas_dir, "region_mapping", f"{map_v}.json"),
        # ELASTIX PARAM FILES ORIGIN
        "affine": os.path.join(atlas_dir, "elastix_params", "align_affine.txt"),
        "bspline": os.path.join(atlas_dir, "elastix_params", "align_bspline.txt"),
    }


def get_proj_fp_dict(proj_dir):
    return {
        # CONFIGS
        "config_params": os.path.join(proj_dir, "config_params.json"),
        # MY ATLAS AND ELASTIX PARAMS FILES
        "ref": os.path.join(proj_dir, "registration", "0a_reference.tif"),
        "annot": os.path.join(proj_dir, "registration", "0b_annotation.tif"),
        "map": os.path.join(proj_dir, "registration", "0c_mapping.json"),
        "affine": os.path.join(proj_dir, "registration", "0d_align_affine.txt"),
        "bspline": os.path.join(proj_dir, "registration", "0e_align_bspline.txt"),
        # RAW IMG FILE
        "raw": os.path.join(proj_dir, "raw.zarr"),
        # REGISTRATION PROCESSING FILES
        "downsmpl1": os.path.join(proj_dir, "registration", "1_downsmpl1.tif"),
        "downsmpl2": os.path.join(proj_dir, "registration", "2_downsmpl2.tif"),
        "trimmed": os.path.join(proj_dir, "registration", "3_trimmed.tif"),
        "regresult": os.path.join(proj_dir, "registration", "4_regresult.tif"),
        # WHOLE MASK
        "premask_blur": os.path.join(proj_dir, "mask", "1_premask_blur.tif"),
        "mask": os.path.join(proj_dir, "mask", "2_mask_trimmed.tif"),
        "outline": os.path.join(proj_dir, "mask", "3_outline_reg.tif"),
        "mask_reg": os.path.join(proj_dir, "mask", "4_mask_reg.tif"),
        "mask_counts_df": os.path.join(proj_dir, "mask", "5_mask_counts.parquet"),
        # CELL COUNTING ARR FILES
        "overlap": os.path.join(proj_dir, "cellcount", "0_overlap.zarr"),
        "bgrm": os.path.join(proj_dir, "cellcount", "1_bgrm.zarr"),
        "dog": os.path.join(proj_dir, "cellcount", "2_dog.zarr"),
        "adaptv": os.path.join(proj_dir, "cellcount", "3_adaptv.zarr"),
        "threshd": os.path.join(proj_dir, "cellcount", "4_threshd.zarr"),
        "threshd_sizes": os.path.join(proj_dir, "cellcount", "5_threshd_sizes.zarr"),
        "threshd_filt": os.path.join(proj_dir, "cellcount", "6_threshd_filt.zarr"),
        "maxima": os.path.join(proj_dir, "cellcount", "7_maxima.zarr"),
        "wshed_sizes": os.path.join(proj_dir, "cellcount", "8_wshed_sizes.zarr"),
        "wshed_filt": os.path.join(proj_dir, "cellcount", "9_wshed_filt.zarr"),
        "threshd_final": os.path.join(proj_dir, "cellcount", "10_threshd_f.zarr"),
        "maxima_final": os.path.join(proj_dir, "cellcount", "10_maxima_f.zarr"),
        "wshed_final": os.path.join(proj_dir, "cellcount", "10_wshed_f.zarr"),
        # CELL COUNTING DF FILES
        "maxima_df": os.path.join(proj_dir, "analysis", "11_maxima.parquet"),
        "cells_raw_df": os.path.join(proj_dir, "analysis", "11_cells_raw.parquet"),
        "cells_trfm_df": os.path.join(proj_dir, "analysis", "12_cells_trfm.parquet"),
        "cells_df": os.path.join(proj_dir, "analysis", "13_cells.parquet"),
        "cells_agg_df": os.path.join(proj_dir, "analysis", "14_cells_agg.parquet"),
        "cells_agg_csv": os.path.join(proj_dir, "analysis", "15_cells_agg.csv"),
        # VISUAL CHECK FROM CELL DF FILES
        "points_check": os.path.join(proj_dir, "visual_check", "points.zarr"),
        "heatmap_check": os.path.join(proj_dir, "visual_check", "heatmap.zarr"),
        "points_trfm_check": os.path.join(proj_dir, "visual_check", "points_trfm.zarr"),
        "heatmap_trfm_check": os.path.join(
            proj_dir, "visual_check", "heatmap_trfm.zarr"
        ),
    }


def make_proj_dirs(proj_dir):
    os.makedirs(os.path.join(proj_dir, "registration"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "cellcount"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "visual_check"), exist_ok=True)


def init_params(proj_fp_dict, **kwargs):
    # Making registration params json
    try:  # If file exists
        rp = ConfigParamsModel.model_validate(read_json(proj_fp_dict["config_params"]))
    except FileNotFoundError as e:  # If file does not exist
        logging.info(e)
        logging.info("Making new params json")
        rp = ConfigParamsModel()
    # If there are any updates
    if kwargs != {}:
        # Update registration params json
        rp = rp.model_validate(rp.model_copy(update=kwargs))
        # Writing registration params json
        write_json(proj_fp_dict["config_params"], rp.model_dump())
    # Returning the registration params
    return rp
