import dask.dataframe as dd
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.funcs.visual_check_funcs_dask import (
    coords2heatmaps,
    coords2points,
)
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images/P11_agg_2.5x_1xzoom_02072024"

    pfm = get_proj_fp_model(proj_dir)

    with cluster_proc_contxt(LocalCluster()):
        # df = dd.read_parquet(pfm["cells_raw_df"]).compute()
        # coords2heatmaps(
        #     df,
        #     5,
        #     da.from_zarr(pfm["raw"]).shape,
        #     pfm["heatmap_check"],
        # )

        # df = dd.read_parquet(pfm["cells_raw_df"]).compute()
        # coords2points(
        #     df,
        #     da.from_zarr(pfm["raw"]).shape,
        #     pfm["points_check"],
        # )

        df = dd.read_parquet(pfm["cells_trfm_df"])
        coords2heatmaps(
            df,
            3,
            tifffile.imread(pfm["ref"]).shape,
            pfm["heatmap_trfm_check"],
        )

        df = dd.read_parquet(pfm["cells_trfm_df"])
        coords2points(
            df,
            tifffile.imread(pfm["ref"]).shape,
            pfm["points_trfm_check"],
        )
