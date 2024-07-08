import os

import dask.array as da
import tifffile
from dask.distributed import Client, LocalCluster


if __name__ == "__main__":
    # Filenames
    # in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # Out filenames
    raw_arr_fp = os.path.join(out_dir, "raw.zarr")
    
    # Defining process params
    chunks = (50, 50, 50)

    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client.dashboard_link)

    # Tiff to zarr (better without cluster)
    shape = tifffile.memmap(in_fp).shape
    hshape = [int(i / 2) for i in shape]
    arr_raw = da.from_array(
        # tifffile.memmap(in_fp),
        # tifffile.memmap(in_fp)[: hshape[0], : hshape[1], : hshape[2]],
        tifffile.memmap(in_fp)[::2, ::2, ::2],
        chunks=chunks,
    )
    arr_raw.to_zarr(raw_arr_fp, overwrite=True)

    client.close()