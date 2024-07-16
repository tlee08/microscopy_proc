import os

import tifffile
from dask.distributed import Client, LocalCluster

# logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/large_cellcount/raw.zarr"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    reg_dir = os.path.join(out_dir, "registration")

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
    z_trim = slice(None, -5)
    y_trim = slice(60, -50)
    x_trim = slice(None, None)

    os.makedirs(reg_dir, exist_ok=True)

    # Making Dask cluster and client
    # cluster = LocalCluster(n_workers=8, threads_per_worker=2)
    cluster = LocalCluster()
    client = Client(cluster)

    # Rough downsample
    # arr_raw = da.from_zarr(in_fp)
    # arr_downsmpl1 = downsmpl_rough_arr(arr_raw, z_rough, y_rough, x_rough).compute()
    # tifffile.imwrite(os.path.join(reg_dir, "1_downsmpl1.tif"), arr_downsmpl1)

    # arr_downsmpl1 = tifffile.imread(os.path.join(reg_dir, "1_downsmpl1.tif"))
    # Fine downsample
    # arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, z_fine, y_fine, x_fine)
    # tifffile.imwrite(os.path.join(reg_dir, "2_downsmpl2.tif"), arr_downsmpl2)

    arr_downsmpl2 = tifffile.imread(os.path.join(reg_dir, "2_downsmpl2.tif"))
    # Trim
    arr_trimmed = arr_downsmpl2[z_trim, y_trim, x_trim]
    tifffile.imwrite(os.path.join(reg_dir, "3_trimmed.tif"), arr_trimmed)

    # Closing client
    client.close()
