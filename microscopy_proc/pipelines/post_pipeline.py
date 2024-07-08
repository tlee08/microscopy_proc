import os

import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.post_funcs import coords_to_heatmaps

if __name__ == "__main__":
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Defining process params
    n_workers = 5
    depth = 50
    chunks = (500, 1400, 1400)
    device_id = 1

    # Making Dask cluster and client
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    # region_df = dd.read_parquet(os.path.join(out_dir, "9_regions.parquet")).compute()
    # coords_to_points(
    #     region_df,
    #     da.from_zarr(os.path.join(out_dir, "raw.zarr")).shape,
    #     os.path.join(out_dir, "points.zarr"),
    # )

    maxima_df = dd.read_parquet(os.path.join(out_dir, "9_maximas.parquet")).compute()
    coords_to_heatmaps(
        maxima_df,
        5,
        da.from_zarr(os.path.join(out_dir, "raw.zarr")).shape,
        os.path.join(out_dir, "heatmaps.zarr"),
    )

    # Closing client
    client.close()
