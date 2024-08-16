import json

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from microscopy_proc.funcs.elastix_funcs import transformation_coords
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs as Gf
from microscopy_proc.funcs.map_funcs import df_map_ids, nested_tree_dict_to_df
from microscopy_proc.funcs.mask_funcs import (
    fill_outline,
    make_outline,
    mask_to_region_counts,
)
from microscopy_proc.funcs.visual_check_funcs import coords_to_points

# logging.basicConfig(level=logging.DEBUG)
# from prefect import flow
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    make_proj_dirs,
)
from microscopy_proc.viewer.image_viewer import view_imgs

if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

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
    coords_to_points(
        outline_df[outline_df["is_in"] == 1],
        arr_ref.shape,
        proj_fp_dict["outline"],
    )
    outline_in = tifffile.imread(proj_fp_dict["outline"])
    coords_to_points(
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
        annot_df = nested_tree_dict_to_df(json.load(f)["msg"][0])
    # Getting the annotation name for every cell (zyx coord)
    mask_counts_df = pd.merge(
        left=mask_to_region_counts(np.full(arr_annot.shape, 1), arr_annot),
        right=mask_to_region_counts(arr_mask_reg, arr_annot),
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

    # View images
    view_imgs(
        [
            proj_fp_dict["ref"],
            # proj_fp_dict["trimmed"],
            # proj_fp_dict["smoothed"],
            # proj_fp_dict["mask"],
            proj_fp_dict["outline"],
            # proj_fp_dict["outline_reg"],
            proj_fp_dict["mask_reg"],
        ],
        [5, 5, 5, 5, 5, 5],
        [slice(None, None), slice(None, None), slice(None, None)],
    )
