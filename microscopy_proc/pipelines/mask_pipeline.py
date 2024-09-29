import json
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from microscopy_proc.funcs.elastix_funcs import transformation_coords
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs as Gf
from microscopy_proc.funcs.map_funcs import combine_nested_regions, nested_tree_dict2df
from microscopy_proc.funcs.mask_funcs import (
    fill_outline,
    make_outline,
    mask2region_counts,
)
from microscopy_proc.funcs.visual_check_funcs import coords2points
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    make_proj_dirs,
)


def make_mask_for_ref(
    pfm: dict,
    **kwargs,
):
    """
    Makes mask of actual image in reference space.
    Also stores # and proportion of existent voxels
    for each region.
    """
    # Update registration params json
    rp = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading ref and trimmed imgs
    arr_ref = tifffile.imread(pfm.ref)
    arr_trimmed = tifffile.imread(pfm.trimmed)
    # Making mask
    arr_blur = Gf.gauss_blur_filt(arr_trimmed, rp.mask_gaus_blur)
    tifffile.imwrite(pfm.premask_blur, arr_blur)
    arr_mask = Gf.manual_thresh(arr_blur, rp.mask_thresh)
    tifffile.imwrite(pfm.mask, arr_mask)

    # Make outline
    outline_df = make_outline(arr_mask)
    # Transformix on coords
    outline_df[.z", "y", "x] = (
        transformation_coords(
            outline_df,
            pfm.ref,
            pfm.regresult,
        )[.z", "y", "x]
        .round(0)
        .astype(np.int32)
    )
    # Filtering out of bounds coords
    outline_df = outline_df.query(
        f"z >= 0 and z < {arr_ref.shape[0]} and y >= 0 and y < {arr_ref.shape[1]} and x >= 0 and x < {arr_ref.shape[2]}"
    )

    # Make outline img (1 for in, 2 for out)
    coords2points(
        outline_df[outline_df.is_in == 1],
        arr_ref.shape,
        pfm.outline,
    )
    outline_in = tifffile.imread(pfm.outline)
    coords2points(
        outline_df[outline_df.is_in == 0],
        arr_ref.shape,
        pfm.outline,
    )
    outline_out = tifffile.imread(pfm.outline)
    tifffile.imwrite(pfm.outline, outline_in + outline_out * 2)

    # Fill in outline to recreate mask (not perfect)
    arr_mask_reg = fill_outline(arr_ref, outline_df)
    # Opening (removes FP) and closing (fills FN)
    arr_mask_reg = ndimage.binary_closing(arr_mask_reg, iterations=2).astype(np.uint8)
    arr_mask_reg = ndimage.binary_opening(arr_mask_reg, iterations=2).astype(np.uint8)
    # Saving
    tifffile.imwrite(pfm.mask_reg, arr_mask_reg)

    # Counting mask voxels in each region
    arr_annot = tifffile.imread(pfm.annot)
    with open(pfm.map, "r") as f:
        annot_df = nested_tree_dict2df(json.load(f).msg[0])
    # Getting the annotation name for every cell (zyx coord)
    mask_counts_df = pd.merge(
        left=mask2region_counts(np.full(arr_annot.shape, 1), arr_annot),
        right=mask2region_counts(arr_mask_reg, arr_annot),
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("_annot", "_mask"),
    ).fillna(0)
    # Combining (summing) the cells_grouped_df values for parent regions using the annot_df
    mask_counts_df = combine_nested_regions(mask_counts_df, annot_df)
    # Calculating proportion of mask volume in each region
    mask_counts_df.volume_prop = (
        mask_counts_df.volume_mask / mask_counts_df.volume_annot
    )
    # Saving
    mask_counts_df.to_parquet(pfm.mask_counts_df)

    # # View images
    # view_imgs(
    #     [
    #         pfm.ref,
    #         # pfm.trimmed,
    #         # pfm.smoothed,
    #         # pfm.mask,
    #         pfm.outline,
    #         # pfm.outline_reg,
    #         pfm.mask_reg,
    #     ],
    #     [5, 5, 5, 5, 5, 5],
    #     [slice(None, None), slice(None, None), slice(None, None)],
    # )


if __name__ == "__main__":
    # # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # # Getting file paths
    # pfm = get_proj_fp_model(proj_dir)
    # Making project folders
    # make_proj_dirs(proj_dir)
    # # Running mask pipeline
    # make_mask_for_ref(pfm)

    # Filenames
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"

    for i in os.listdir(batch_proj_dir):
        # Only running specific files
        if i not in [
            "G3_agg_2.5x_1xzoom_03072024",
        ]:
            continue
        # Logging which file is being processed
        print(f"Running: {i}")
        try:
            # Filenames
            proj_dir = os.path.join(batch_proj_dir, i)
            # Getting file paths
            pfm = get_proj_fp_model(proj_dir)

            # Making project folders
            make_proj_dirs(os.path.join(batch_proj_dir, i))

            # Running mask pipeline
            make_mask_for_ref(pfm)
            print()
        except Exception as e:
            print(f"Error: {e}")
            continue
