import os

import cupy as cp
import dask.array as da
import numpy as np
import tifffile
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.cellc_funcs import region_to_coords
from microscopy_proc.funcs.dask_funcs import (
    block_to_coords,
    get_maxima_block_from_region,
    get_region_block,
)

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/cellcount/cropped abcd_larger.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    # Out filenames
    region_arr_fp = os.path.join(out_dir, "regions.zarr")
    maxima_arr_fp = os.path.join(out_dir, "maxima.zarr")
    region_df_fp = os.path.join(out_dir, "regions.parquet")
    maxima_df_fp = os.path.join(out_dir, "maxima.parquet")

    # Defining process params
    n_workers = 1
    depth = 50
    chunks = (500, 1400, 1400)
    device_id = 1

    # Making Dask cluster and client
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    with cp.cuda.Device(device_id):
        # Loading image into a Dask array
        arr_raw = tifffile.memmap(in_fp).astype(np.uint16)
        # Loading raw img as Dask array
        arr_raw = da.from_array(arr_raw, chunks=chunks)
        # Getting region arr
        arr_region = arr_raw.map_overlap(
            get_region_block, depth=depth, boundary="reflect"
        )
        arr_region = arr_region.rechunk(arr_region.chunksize)
        arr_region.to_zarr(region_arr_fp, overwrite=True)
        arr_region = da.from_zarr(region_arr_fp)
        # Getting maxima arr
        arr_maxima = da.map_overlap(
            get_maxima_block_from_region, arr_raw, arr_region, depth=depth
        )
        arr_maxima = arr_maxima.rechunk(arr_maxima.chunksize)
        arr_maxima.to_zarr(maxima_arr_fp, overwrite=True)
        arr_maxima = da.from_zarr(maxima_arr_fp)

        # Converting from region arr to coords df
        cell_coords = block_to_coords(region_to_coords, arr_region)
        cell_coords.to_parquet(region_df_fp)
        # Converting from maxima arr to coords df
        cell_coords = block_to_coords(region_to_coords, arr_maxima)
        cell_coords.to_parquet(maxima_df_fp)

    # Closing client
    client.close()
