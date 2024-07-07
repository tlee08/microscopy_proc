import os

import numpy as np
import tifffile
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.reg_funcs import downsmpl_fine_arr, downsmpl_rough_arr

# logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    # Out filenames
    downs_i_arr_fp = os.path.join(out_dir, "downs_1.tif")
    downs_f_arr_fp = os.path.join(out_dir, "downs_2.tif")
    trimmed_arr_fp = os.path.join(out_dir, "trimmed.tif")

    # Defining params
    z_d_rough = 0.3
    y_d_rough = 0.1
    x_d_rough = 0.1
    z_d_fine = 0.8
    y_d_fine = 0.8
    x_d_fine = 0.8

    # Defining process params
    n_workers = 10
    depth = 50
    chunks = (500, 1400, 1400)
    device_id = 1

    # Making Dask cluster and client
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    # Rough downsample
    arr_raw = tifffile.memmap("/home/linux1/Desktop/A-1-1/abcd.tif")
    arr_downs_i = downsmpl_rough_arr(arr_raw, z_d_rough, y_d_rough, x_d_rough)
    # Converting from memmap to np array in memory
    arr_downs_i = np.asarray(arr_downs_i)
    arr_downs_f = downsmpl_fine_arr(arr_downs_i, z_d_fine, y_d_fine, x_d_fine)
    # Trim
    # TODO
    arr_trimmed = arr_downs_f
    tifffile.imwrite(trimmed_arr_fp, arr_trimmed)

    # Closing client
    client.close()
