import gc
import os

import dask.array as da
import numpy as np
import tifffile
from dask.distributed import Client, LocalCluster

from microscopy_proc.constants import INTER_RAW_CHUNKS, RAW_CHUNKS
from microscopy_proc.utils.io_utils import silentremove


def arr_to_zarr_chunk(arr, out_fp):
    da.from_array(arr, chunks=RAW_CHUNKS).to_zarr(out_fp, overwrite=True)


def inter_fp(fp, idx):
    return f"{fp}.{'.'.join((str(i) for i in idx))}.zarr"


def tiff_to_zarr_ijk_chunk(in_fp, out_fp, idx):
    idx = np.array(idx)
    # Getting tiff array shape
    shape = tifffile.memmap(in_fp).shape
    # Getting chunk dimensions
    lb = idx * INTER_RAW_CHUNKS
    ub = np.minimum(lb + INTER_RAW_CHUNKS, shape)
    # Writing chunked tiff to zarr
    arr_to_zarr_chunk(
        tifffile.memmap(in_fp)[*(slice(i, j) for i, j in zip(lb, ub))],
        inter_fp(out_fp, idx),
    )
    # Freeing memory
    gc.collect()
    # Returning zarr array
    return da.from_zarr(inter_fp(out_fp, idx))


def tiff_to_zarr_chunks(in_fp, out_fp):
    # Getting shape
    shape = np.array(tifffile.memmap(in_fp).shape)
    # Making array of smaller zarr files (Futures)
    chunks_shape = [int(np.ceil(i / j)) for i, j in zip(shape, INTER_RAW_CHUNKS)]
    arr_c = np.zeros(chunks_shape, dtype=object)
    # Making chunked tiff to intermediate zarr files
    grid = np.meshgrid(*(np.arange(i) for i in chunks_shape), indexing="ij")
    for idx in zip(*(i.ravel() for i in grid)):
        # Writing chunked tiff to zarr
        arr_c[*idx] = tiff_to_zarr_ijk_chunk(in_fp, out_fp, idx)
    return arr_c


def remove_intermediate_zarr_files(chunks_shape, out_fp):
    grid = np.meshgrid(*(np.arange(i) for i in chunks_shape), indexing="ij")
    for idx in zip(*(i.ravel() for i in grid)):
        silentremove(inter_fp(out_fp, idx))


def large_tiff_to_zarr(in_fp, out_fp):
    # Making chunked tiff to intermediate zarr files
    arr_c = tiff_to_zarr_chunks(in_fp, out_fp)
    # Combining intermediate zarr files
    arr = da.block(arr_c.tolist())
    arr.to_zarr(out_fp, overwrite=True)
    # Removing intermediate zarr files
    remove_intermediate_zarr_files(arr_c.shape, out_fp)


if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Making Dask cluster and client (thread-based cluster)
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client.dashboard_link)

    large_tiff_to_zarr(in_fp, os.path.join(out_dir, "raw.zarr"))

    # Closing client
    client.close()
