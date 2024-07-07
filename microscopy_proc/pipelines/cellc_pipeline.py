import cupy as cp
import dask
import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np
import tifffile
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.dask_funcs import (
    calc_inds,
    get_maxima_block_from_region,
    get_maxima_df_block,
    get_region_block,
)

# logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    # Defining process params
    n_workers = 10
    depth = 50
    chunks = (500, 1400, 1400)
    device_id = 1

    # Making Dask cluster and client
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    # x = tifffile.memmap("/home/linux1/Desktop/A-1-1/abcd.tif")
    # x_d = da.from_array(x, chunks=(300, 1400, 1400))
    # x_s = x_d.map_blocks(lambda x: zoom_arr(x, 0.2), dtype=np.uint16)
    # x_c = x_s.compute()
    # tifffile.imwrite("downs.tif", x_c)

    with cp.cuda.Device(device_id):
        # Loading image into a Dask array
        arr_raw = tifffile.memmap("cropped abcd_larger.tif").astype(np.uint16)
        # Loading raw img as Dask array
        arr_raw = da.from_array(arr_raw, chunks=chunks)
        # Getting regions arr
        arr_region = arr_raw.map_overlap(
            get_region_block, depth=depth, boundary="reflect"
        )
        arr_region = arr_region.rechunk(arr_region.chunksize)
        arr_region.to_zarr("regions.zarr", overwrite=True)
        arr_region = da.from_zarr("regions.zarr")
        # Getting maxima arr
        arr_maxima = da.map_overlap(
            get_maxima_block_from_region, arr_raw, arr_region, depth=depth
        )
        arr_maxima = arr_maxima.rechunk(arr_maxima.chunksize)
        arr_maxima.to_zarr("maxima.zarr", overwrite=True)
        arr_maxima = da.from_zarr("maxima.zarr")

        # Converting from regions arr to coords df
        inds = calc_inds(arr_maxima)
        dd.from_delayed(
            [
                dask.delayed(get_maxima_df_block)(block, i, j, k)
                for block, i, j, k in zip(
                    arr_maxima.to_delayed().ravel(),
                    inds[0].ravel(),
                    inds[1].ravel(),
                    inds[2].ravel(),
                )
            ]
        ).to_parquet("maxima.parquet")

    # Closing client
    client.close()
