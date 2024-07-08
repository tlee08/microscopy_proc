import os

import tifffile
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.tiff_to_zarr_funcs import mmap_to_zarr

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Making Dask cluster and client (thread-based cluster)
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client.dashboard_link)

    shape = tifffile.memmap(in_fp).shape
    hshape = [int(i / 2) for i in shape]
    mmap = tifffile.memmap(in_fp)[::2, ::2, ::2]
    mmap_to_zarr(mmap, os.path.join(out_dir, "raw.zarr"))

    # Closing client
    client.close()
