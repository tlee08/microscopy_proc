import os

import dask.array as da
import tifffile
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from microscopy_proc.funcs.cellc_funcs import region_to_coords
from microscopy_proc.funcs.dask_funcs import (
    block_to_coords,
    get_maxima_block_from_region,
    get_region_block,
)

if __name__ == "__main__":
    # Filenames
    # in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # Out filenames
    raw_arr_fp = os.path.join(out_dir, "raw.zarr")
    region_arr_fp = os.path.join(out_dir, "regions.zarr")
    maxima_arr_fp = os.path.join(out_dir, "maxima.zarr")
    region_df_fp = os.path.join(out_dir, "regions.parquet")
    maxima_df_fp = os.path.join(out_dir, "maxima.parquet")

    # Defining process params
    depth = 50
    chunks = (500, 1000, 1000)

    # Making Dask cluster and client
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client.dashboard_link)

    # Making region arr
    arr_raw = da.from_zarr(raw_arr_fp)
    arr_region = arr_raw.map_overlap(
        get_region_block,
        depth=depth,
        boundary="reflect",
    )
    arr_region = arr_region.rechunk(arr_region.chunksize)
    arr_region.to_zarr(region_arr_fp, overwrite=True)

    # Making maxima arr
    arr_raw = da.from_zarr(raw_arr_fp)
    arr_region = da.from_zarr(region_arr_fp)
    arr_maxima = da.map_overlap(
        get_maxima_block_from_region, arr_raw, arr_region, depth=depth
    )
    arr_maxima = arr_maxima.rechunk(arr_maxima.chunksize)
    arr_maxima.to_zarr(maxima_arr_fp, overwrite=True)

    exit()

    # Converting from region arr to coords df
    cell_coords = block_to_coords(region_to_coords, arr_region)
    cell_coords.to_parquet(region_df_fp)
    # Converting from maxima arr to coords df
    cell_coords = block_to_coords(region_to_coords, arr_maxima)
    cell_coords.to_parquet(maxima_df_fp)

    # Closing client
    client.close()
