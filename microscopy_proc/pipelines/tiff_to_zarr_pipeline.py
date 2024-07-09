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


def make_inter_zarrs(in_fp, out_fp, chunkshape):
    # Making chunked tiff to intermediate zarr files
    grid = np.meshgrid(*(np.arange(i) for i in chunkshape), indexing="ij")
    for idx in zip(*(i.ravel() for i in grid)):
        print(idx)
        tiff_to_zarr_ijk_chunk(in_fp, out_fp, idx)


def get_inter_zarrs(fp, chunkshape):
    # Making array of smaller zarr files
    arr_c = np.zeros(chunkshape, dtype=object)
    # Returning zarr array
    grid = np.meshgrid(*(np.arange(i) for i in chunkshape), indexing="ij")
    for idx in zip(*(i.ravel() for i in grid)):
        arr_c[*idx] = da.from_zarr(inter_fp(fp, idx))
    return arr_c


def remove_inter_zarrs(fp, chunkshape):
    grid = np.meshgrid(*(np.arange(i) for i in chunkshape), indexing="ij")
    for idx in zip(*(i.ravel() for i in grid)):
        silentremove(inter_fp(fp, idx))


def large_tiff_to_zarr(in_fp, out_fp):
    # Getting shape
    shape = np.array(tifffile.memmap(in_fp).shape)
    chunkshape = [int(np.ceil(i / j)) for i, j in zip(shape, INTER_RAW_CHUNKS)]
    # # Making chunked tiff to intermediate zarr files
    make_inter_zarrs(in_fp, out_fp, chunkshape)
    # Getting array of intermediate zarr files
    arr_c = get_inter_zarrs(out_fp, chunkshape)
    # Combining intermediate zarr files
    arr = da.block(arr_c.tolist())
    arr.to_zarr(out_fp, overwrite=True)
    # Removing intermediate zarr files
    remove_inter_zarrs(arr_c.shape, out_fp)


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
