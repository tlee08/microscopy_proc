import json
import logging
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from microscopy_proc.funcs.elastix_funcs import transformation_coords
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs as Gf
from microscopy_proc.funcs.map_funcs import df_map_ids, nested_tree_dict2df
from microscopy_proc.funcs.mask_funcs import (
    fill_outline,
    make_outline,
    mask2region_counts,
)
from microscopy_proc.funcs.visual_check_funcs import coords2points
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    make_proj_dirs,
)


def make_mask_for_ref(proj_fp_dict: dict):
    """
    Makes mask of actual image in reference space.
    Also stores # and proportion of existent voxels
    for each region.
    """
    # Reading ref and trimmed imgs
    arr_ref = tifffile.imread(proj_fp_dict["ref"])
    arr_trimmed = tifffile.imread(proj_fp_dict["trimmed"])

    # Making mask
    arr_smoothed = Gf.gauss_blur_filt(arr_trimmed, 1)
    arr_mask = Gf.manual_thresh(arr_smoothed, 400)
    tifffile.imwrite(proj_fp_dict["mask"], arr_mask)

    # Make outline
    outline_df = make_outline(arr_mask)
    # Transformix on coords
    outline_df[["z", "y", "x"]] = (
        transformation_coords(
            outline_df,
            proj_fp_dict["ref"],
            proj_fp_dict["regresult"],
        )[["z", "y", "x"]]
        .round(0)
        .astype(np.int32)
    )
    # Filtering out of bounds coords
    outline_df = outline_df.query(
        f"z >= 0 and z < {arr_ref.shape[0]} and y >= 0 and y < {arr_ref.shape[1]} and x >= 0 and x < {arr_ref.shape[2]}"
    )

    # Make outline img (1 for in, 2 for out)
    coords2points(
        outline_df[outline_df["is_in"] == 1],
        arr_ref.shape,
        proj_fp_dict["outline"],
    )
    outline_in = tifffile.imread(proj_fp_dict["outline"])
    coords2points(
        outline_df[outline_df["is_in"] == 0],
        arr_ref.shape,
        proj_fp_dict["outline"],
    )
    outline_out = tifffile.imread(proj_fp_dict["outline"])
    tifffile.imwrite(proj_fp_dict["outline"], outline_in + outline_out * 2)

    # Fill in outline to recreate mask (not perfect)
    arr_mask_reg = fill_outline(arr_ref, outline_df)
    # Opening (removes FP) and closing (fills FN)
    arr_mask_reg = ndimage.binary_closing(arr_mask_reg, iterations=2).astype(np.uint8)
    arr_mask_reg = ndimage.binary_opening(arr_mask_reg, iterations=2).astype(np.uint8)
    # Saving
    tifffile.imwrite(proj_fp_dict["mask_reg"], arr_mask_reg)

    # Counting mask voxels in each region
    arr_annot = tifffile.imread(proj_fp_dict["annot"])
    with open(proj_fp_dict["map"], "r") as f:
        annot_df = nested_tree_dict2df(json.load(f)["msg"][0])
    # Getting the annotation name for every cell (zyx coord)
    mask_counts_df = pd.merge(
        left=mask2region_counts(np.full(arr_annot.shape, 1), arr_annot),
        right=mask2region_counts(arr_mask_reg, arr_annot),
        how="left",
        on="id",
        suffixes=("_annot", "_mask"),
    ).fillna(0)
    mask_counts_df["volume_prop"] = (
        mask_counts_df["volume_mask"] / mask_counts_df["volume_annot"]
    )
    mask_counts_df = df_map_ids(mask_counts_df, annot_df)
    # Saving
    mask_counts_df.to_parquet(proj_fp_dict["mask_counts_df"])

    # # View images
    # view_imgs(
    #     [
    #         proj_fp_dict["ref"],
    #         # proj_fp_dict["trimmed"],
    #         # proj_fp_dict["smoothed"],
    #         # proj_fp_dict["mask"],
    #         proj_fp_dict["outline"],
    #         # proj_fp_dict["outline_reg"],
    #         proj_fp_dict["mask_reg"],
    #     ],
    #     [5, 5, 5, 5, 5, 5],
    #     [slice(None, None), slice(None, None), slice(None, None)],
    # )


if __name__ == "__main__":
    # # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # # Getting file paths
    # proj_fp_dict = get_proj_fp_dict(proj_dir)
    # Making project folders
    # make_proj_dirs(proj_dir)
    # # Running mask pipeline
    # make_mask_for_ref(proj_fp_dict)

    # Filenames
    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    # in_fp_dir and batch_proj_dir cannot be the same

    # for i in os.listdir(batch_fp_dir):
    proj_dir = os.path.join(batch_proj_dir, "P8_2.5x_1x_zoom_07082024")
    # Getting file paths
    proj_fp_dict = get_proj_fp_dict(proj_dir)

    # Checking if it is a directory
    # if not os.path.isdir(proj_dir):
    #     continue
    # Logging which file is being processed
    # logging.info(f"Running: {i}")
    try:
        # Making project folders
        make_proj_dirs(proj_dir)

        # Running mask pipeline
        make_mask_for_ref(proj_fp_dict)
    except Exception as e:
        logging.error(f"Error: {e}")
        # continue
