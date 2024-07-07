import os

import dask.dataframe as dd
import tifffile
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.post_funcs import coords_to_heatmaps, coords_to_points

if __name__ == "__main__":
    # Filenames
    in_img_fp = "/home/linux1/Desktop/A-1-1/cellcount/cropped abcd_larger.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # Out filenames
    regions_arr_fp = os.path.join(out_dir, "regions.zarr")
    maxima_arr_fp = os.path.join(out_dir, "maxima.zarr")
    regions_df_fp = os.path.join(out_dir, "regions.parquet")
    maxima_df_fp = os.path.join(out_dir, "maxima.parquet")

    # Defining process params
    n_workers = 5
    depth = 50
    chunks = (500, 1400, 1400)
    device_id = 1

    # Making Dask cluster and client
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    region_df = dd.read_parquet(regions_df_fp).compute()
    coords_to_points(
        region_df,
        tifffile.memmap(in_img_fp).shape,
        os.path.join(out_dir, "points.tif"),
    )

    maxima_df = dd.read_parquet(maxima_df_fp).compute()
    coords_to_heatmaps(
        maxima_df,
        5,
        tifffile.memmap(in_img_fp).shape,
        os.path.join(out_dir, "heatmaps.tif"),
    )

    # Closing client
    client.close()
