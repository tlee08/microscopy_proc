import os

import dask.array as da
import tifffile
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.reg_funcs import downsmpl_fine_arr, downsmpl_rough_arr

# logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/large_cellcount/raw.zarr"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount/registration"

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
    z_trim = slice(None, None)
    y_trim = slice(None, None)
    x_trim = slice(None, None)

    # Making Dask cluster and client
    # cluster = LocalCluster(n_workers=8, threads_per_worker=2)
    cluster = LocalCluster()
    client = Client(cluster)

    # Rough downsample
    arr_raw = da.from_zarr(in_fp)
    arr_downsmpl1 = downsmpl_rough_arr(arr_raw, z_rough, y_rough, x_rough).compute()
    tifffile.imwrite(os.path.join(out_dir, "downsmpl1.tif"), arr_downsmpl1)

    # arr_downsmpl1 = tifffile.imread(os.path.join(out_dir, "downsmpl_1.tif"))
    # Fine downsample
    arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, z_fine, y_fine, x_fine)
    tifffile.imwrite(os.path.join(out_dir, "downsmpl2.tif"), arr_downsmpl2)

    # arr_downsmpl2 = tifffile.imread(os.path.join(out_dir, "downsmpl2.tif"))
    # Trim
    arr_trimmed = arr_downsmpl2[z_trim, y_trim, x_trim]
    tifffile.imwrite(os.path.join(out_dir, "trimmed.tif"), arr_trimmed)

    # Getting atlas and transformation files

    # Registration
    registration(
        fixed_img_fp=for_registration_fp,
        moving_img_fp=ref_fp,
        output_img_fp=registration_result_fp,
        affine_fp=affine_fp,
        bspline_fp=bspline_fp,
    )

    # Closing client
    client.close()
