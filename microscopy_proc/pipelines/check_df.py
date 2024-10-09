import logging

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
)

# logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL)

if __name__ == "__main__":
    # Filenames
    in_fp = (
        "/home/linux1/Desktop/Sample_11_zoom0.52_2.5x_dual_side_fusion_2x4 vertical tif"
    )
    proj_dir = "/home/linux1/Desktop/example_proj"
    # in_fp = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX Aggression cohort 1 stitched TIF images for analysis"
    # proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    # exp_name = "G17_2.5x_1x_zoom_07082024"
    # in_fp = os.path.join(in_fp, exp_name)
    # proj_dir = os.path.join(proj_dir, exp_name)

    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    pfm = get_proj_fp_model(proj_dir)

    # Set slicing
    slicer = [
        slice(600, 650, None),
        slice(1400, 3100, None),
        slice(500, 3100, None),
    ]
    # Reading maxima from arr
    maxima_arr = da.from_zarr(pfm.maxima_final)
    maxima_arr = maxima_arr[*slicer].compute()
    maxima_arr_df = pd.DataFrame(np.where(maxima_arr), columns=["z", "y", "x"])
    # Reading cells df
    cells_raw_df = dd.read_parquet(pfm.cells_raw_df).compute()
    # Reading points
    points_arr = da.from_zarr(pfm.points_check)
    points_arr = points_arr[*slicer].compute()
    points_arr_df = pd.DataFrame(np.where(points_arr), columns=["z", "y", "x"])
