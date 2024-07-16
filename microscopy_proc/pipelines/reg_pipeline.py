import os

import dask.array as da
from dask.distributed import Client, LocalCluster

from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.funcs.reg_funcs import downsmpl_rough_arr
from microscopy_proc.utils.dask_utils import disk_cache

# logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # Out filenames
    downs_i_arr_fp = os.path.join(out_dir, "downsmpl_1.tif")
    downs_f_arr_fp = os.path.join(out_dir, "downsmpl_2.tif")
    trimmed_arr_fp = os.path.join(out_dir, "trimmed.tif")

    # Defining params
    # Rough scale
    z_rough = 0.3
    y_rough = 0.1
    x_rough = 0.1
    # Fine scale
    z_fine = 0.8
    y_fine = 0.8
    x_fine = 0.8
    # Trimming
    z_trim = slice(100, 400)
    y_trim = slice(1000, 5000)
    x_trim = slice(1000, 5000)

    # Making Dask cluster and client
    # cluster = LocalCluster(n_workers=8, threads_per_worker=2)
    cluster = LocalCluster()
    client = Client(cluster)

    # Rough downsample
    # arr_raw = tifffile.memmap("/home/linux1/Desktop/A-1-1/abcd.tif")

    arr_raw = da.from_zarr(os.path.join(out_dir, "raw.zarr"), chunks=PROC_CHUNKS)
    arr_downsmpl1 = downsmpl_rough_arr(arr_raw, z_rough, y_rough, x_rough)
    arr_downsmpl1 = disk_cache(arr_downsmpl1, os.path.join(out_dir, "downsmpl_1.zarr"))
    # arr_downsmpl1 = arr_downsmpl1.compute()

    # # Fine downsample
    # arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, z_fine, y_fine, x_fine)
    # arr_downsmpl2 = disk_cache(arr_downsmpl2, os.path.join(out_dir, "downsmpl_2.zarr"))

    # # Trim
    # # TODO
    # arr_trimmed = arr_downsmpl2[z_trim, y_trim, x_trim]

    # # Saving
    # tifffile.imwrite(downs_i_arr_fp, arr_downsmpl1)

    # # Closing client
    # client.close()
