import os

import numpy as np
import tifffile
from natsort import natsorted

from microscopy_proc.pipelines.pipeline_funcs import (
    coords2heatmap_trfm_pipeline,
    coords2points_trfm_pipeline,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
)

# TODO implement propper export function

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

        coords2points_trfm_pipeline(pfm)
        coords2heatmap_trfm_pipeline(pfm)

        # # Exporting
        # os.makedirs(os.path.join(out_dir, i), exist_ok=True)
        # # trimmed
        # arr = tifffile.imread(pfm.trimmed)
        # tifffile.imwrite(os.path.join(out_dir, i, "trimmed.tif"), arr)
        # # regresult
        # arr = tifffile.imread(pfm.regresult)
        # tifffile.imwrite(os.path.join(out_dir, i, "regresult.tif"), arr)
        # # raw
        # arr = da.from_zarr(pfm.raw)[*trimmer].compute()
        # tifffile.imwrite(os.path.join(out_dir, i, "raw.tif"), arr)
        # # maxima_final
        # arr = da.from_zarr(pfm.maxima_final)[*trimmer].compute()
        # tifffile.imwrite(os.path.join(out_dir, i, "maxima_final.tif"), arr)
        # # wshed_final
        # arr = da.from_zarr(pfm.wshed_final)[*trimmer].compute()
        # tifffile.imwrite(os.path.join(out_dir, i, "wshed_final.tif"), arr)
        # ref
        arr = tifffile.imread(pfm.ref)
        tifffile.imwrite(os.path.join(out_dir, i, "ref.tif"), arr)
        # annot
        arr = tifffile.imread(pfm.annot)
        tifffile.imwrite(os.path.join(out_dir, i, "annot.tif"), arr)
        # coords_trfm
        arr = tifffile.imread(pfm.points_trfm)
        tifffile.imwrite(os.path.join(out_dir, i, "points_trfm.tif"), arr)

        # COMBINING ARRAYS (ZYXC)
        # Combining reg
        # arr1 = (
        #     tifffile.imread(os.path.join(out_dir, i, "trimmed.tif"))
        #     .round(0)
        #     .astype(np.uint16)
        # )
        # arr2 = (
        #     tifffile.imread(os.path.join(out_dir, i, "regresult.tif"))
        #     .round(0)
        #     .astype(np.uint16)
        # )
        # arr = np.stack([arr1, arr2], axis=-1, dtype=np.uint16)
        # tifffile.imwrite(os.path.join(out_dir, i, "combined_reg.tif"), arr)
        # # Combining cellc
        # arr1 = (
        #     tifffile.imread(os.path.join(out_dir, i, "raw.tif"))
        #     .round(0)
        #     .astype(np.uint16)
        # )
        # arr2 = (
        #     tifffile.imread(os.path.join(out_dir, i, "maxima_final.tif"))
        #     .round(0)
        #     .astype(np.uint16)
        # )
        # arr3 = (
        #     tifffile.imread(os.path.join(out_dir, i, "wshed_final.tif"))
        #     .round(0)
        #     .astype(np.uint16)
        # )
        # arr = np.stack([arr1, arr2, arr3], axis=-1, dtype=np.uint16)
        # tifffile.imwrite(os.path.join(out_dir, i, "combined_cellc.tif"), arr)
        # # Combining transformed points
        arr1 = (
            tifffile.imread(os.path.join(out_dir, i, "ref.tif"))
            .round(0)
            .astype(np.uint16)
        )
        arr2 = (
            tifffile.imread(os.path.join(out_dir, i, "annot.tif"))
            .round(0)
            .astype(np.uint16)
        )
        arr3 = (
            tifffile.imread(os.path.join(out_dir, i, "points_trfm.tif"))
            .round(0)
            .astype(np.uint16)
        )
        arr = np.stack([arr1, arr2, arr3], axis=-1, dtype=np.uint16)
        tifffile.imwrite(os.path.join(out_dir, i, "combined_points.tif"), arr)
