# import dask.array as da
import dask.dataframe as dd
import tifffile
from dask.distributed import Client, LocalCluster

# from microscopy_proc.funcs.post_funcs import coords_to_heatmaps, coords_to_points
from microscopy_proc.funcs.visual_check_funcs_dask import (
    coords_to_heatmaps,
    coords_to_points,
)
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict

if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)

    # Making Dask cluster and client
    cluster = LocalCluster()
    client = Client(cluster)
    print(client.dashboard_link)

    # region_df = dd.read_parquet(proj_fp_dict["region_df"])
    # coords_to_points(
    #     region_df,
    #     da.from_zarr(proj_fp_dict["raw"]).shape,
    #     proj_fp_dict["points_check"],
    # )

    # maxima_df = dd.read_parquet(proj_fp_dict["maxima_df"])
    # coords_to_heatmaps(
    #     maxima_df,
    #     5,
    #     da.from_zarr(proj_fp_dict["raw"]).shape,
    #     proj_fp_dict["heatmap_check"],
    # )

    maxima_df = dd.read_parquet(proj_fp_dict["maxima_trfm_df"])
    coords_to_points(
        maxima_df,
        tifffile.imread(proj_fp_dict["ref"]).shape,
        proj_fp_dict["points_trfm_check"],
    )

    maxima_df = dd.read_parquet(proj_fp_dict["maxima_trfm_df"])
    coords_to_heatmaps(
        maxima_df,
        2,
        tifffile.imread(proj_fp_dict["ref"]).shape,
        proj_fp_dict["heatmap_trfm_check"],
    )

    # Closing client
    client.close()
