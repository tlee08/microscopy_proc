import os

import dask.array as da
import dask.dataframe as dd
import tifffile
from dask.distributed import LocalCluster
from natsort import natsorted

from microscopy_proc.funcs.visual_check_funcs_dask import (
    coords2points,
)
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    exp_ls = natsorted(os.listdir(batch_proj_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(batch_proj_dir, i))]
    proj_dir = os.path.join(batch_proj_dir, exp_ls[0])

    pfm = get_proj_fp_model(proj_dir)

    with cluster_proc_contxt(LocalCluster()):
        # coords2heatmaps(
        #     dd.read_parquet(pfm.cells_raw_df).compute(),
        #     5,
        #     da.from_zarr(pfm.raw).shape,
        #     pfm.heatmap_check,
        # )

        coords2points(
            dd.read_parquet(pfm.cells_raw_df).compute(),
            da.from_zarr(pfm.raw).shape,
            pfm.points_check,
        )

        # coords2heatmaps(
        #     dd.read_parquet(pfm.cells_trfm_df),
        #     3,
        #     tifffile.imread(pfm.ref).shape,
        #     pfm.heatmap_trfm_check,
        # )

        coords2points(
            dd.read_parquet(pfm.cells_trfm_df),
            tifffile.imread(pfm.ref).shape,
            pfm.points_trfm_check,
        )
