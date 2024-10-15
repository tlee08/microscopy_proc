import os

import numpy as np
import tifffile
from natsort import natsorted

from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
)

if __name__ == "__main__":
    # Filenames
    root_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    out_dir = "/home/linux1/Desktop/imgs_for_nick"

    overwrite = False

    # Get all experiments
    exp_ls = natsorted(os.listdir(root_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(root_dir, i))]

    for i in exp_ls:
        # Only given files
        if i not in [
            "B3_2.5x_1x_zoom_08082024",
            "G1_reimage_agg_2.5x_1xzoom_05072024",
            "G17_2.5x_1x_zoom_07082024",
            "P8_2.5x_1x_zoom_07082024",
            "R14_agg_2.5x_1xzoom_02072024",
        ]:
            continue
        proj_dir = os.path.join(root_dir, i)
        pfm = get_proj_fp_model(proj_dir)

        trimmer = (
            slice(750, 760, None),
            slice(None, None, None),
            slice(None, None, None),
            # slice(None, None, None),
            # slice(None, None, None),
            # slice(None, None, None),
        )

        # # Exporting
        # os.makedirs(os.path.join(out_dir, i), exist_ok=True)
        # # trimmed
        # arr = tifffile.imread(pfm.trimmed)
        # tifffile.imwrite(os.path.join(out_dir, i, "trimmed"), arr)
        # # regresult
        # arr = tifffile.imread(pfm.regresult)
        # tifffile.imwrite(os.path.join(out_dir, i, "regresult"), arr)
        # # raw
        # arr = da.from_zarr(pfm.raw)[*trimmer].compute()
        # tifffile.imwrite(os.path.join(out_dir, i, "raw"), arr)
        # # maxima_final
        # arr = da.from_zarr(pfm.maxima_final)[*trimmer].compute()
        # tifffile.imwrite(os.path.join(out_dir, i, "maxima_final"), arr)
        # # wshed_final
        # arr = da.from_zarr(pfm.wshed_final)[*trimmer].compute()
        # tifffile.imwrite(os.path.join(out_dir, i, "wshed_final"), arr)

        # COMBINING ARRAYS (ZYXC)
        # Combining reg
        arr1 = tifffile.imread(pfm.trimmed)
        arr2 = tifffile.imread(pfm.regresult)
        arr = np.stack([arr1, arr2], axis=-1, dtype=np.uint16)
        tifffile.imwrite(os.path.join(out_dir, i, "combined_reg.tif"), arr)
        # Combining cellc
        arr1 = tifffile.imread(pfm.raw)
        arr2 = tifffile.imread(pfm.maxima_final)
        arr3 = tifffile.imread(pfm.wshed_final)
        arr = np.stack([arr1, arr2, arr3], axis=-1, dtype=np.uint16)
        tifffile.imwrite(os.path.join(out_dir, i, "combined_cellc.tif"), arr)
