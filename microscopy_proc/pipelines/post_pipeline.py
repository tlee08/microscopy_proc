import os

import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

# from microscopy_proc.funcs.post_funcs import coords_to_heatmaps, coords_to_points
from microscopy_proc.funcs.post_funcs_dask import coords_to_heatmaps, coords_to_points

if __name__ == "__main__":
    # Filenames
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Making Dask cluster and client
    cluster = LocalCluster()
    client = Client(cluster)
    print(client.dashboard_link)

    region_df = dd.read_parquet(os.path.join(out_dir, "10_region.parquet"))
    coords_to_points(
        region_df,
        da.from_zarr(os.path.join(out_dir, "raw.zarr")).shape,
        os.path.join(out_dir, "points.zarr"),
    )

    # maxima_df = dd.read_parquet(os.path.join(out_dir, "10_maxima.parquet")).compute()
    # coords_to_heatmaps(
    #     maxima_df,
    #     5,
    #     da.from_zarr(os.path.join(out_dir, "raw.zarr")).shape,
    #     os.path.join(out_dir, "heatmaps.zarr"),
    # )

    # Closing client
    client.close()
