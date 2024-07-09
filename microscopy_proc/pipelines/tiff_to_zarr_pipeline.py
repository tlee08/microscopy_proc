import os

import dask.array as da
import numpy as np
import tifffile
import zarr

from microscopy_proc.constants import INTER_CHUNKS, PROC_CHUNKS, RAW_CHUNKS
from microscopy_proc.utils.io_utils import silentremove


def inter_fp(fp, idx):
    return f"{fp}.{'.'.join((str(i) for i in idx))}.zarr"


def tiff_to_zarrs_ijk(in_fp, out_fp, idx):
    idx = np.array(idx)
    # Getting tiff memmap
    mmap = tifffile.memmap(in_fp)
    # Getting chunk dimensions
    lb = idx * INTER_CHUNKS
    ub = np.minimum(lb + INTER_CHUNKS, mmap.shape)
    # Writing chunked tiff to zarr
    # da.from_array(
    #     mmap[*(slice(i, j) for i, j in zip(lb, ub))],
    #     chunks=RAW_CHUNKS,
    # ).to_zarr(inter_fp(out_fp, idx), overwrite=True)
    z_f = zarr.open(
        inter_fp(out_fp, idx),
        mode="w",
        shape=tuple(ub - lb),
        dtype=mmap.dtype,
        chunks=RAW_CHUNKS,
    )
    z_f[:] = mmap[*(slice(i, j) for i, j in zip(lb, ub))]
    # Collecting garbage
    del mmap


def make_inter_zarrs(in_fp, out_fp, chunkshape):
    # Making chunked tiff to intermediate zarr files
    grid = np.meshgrid(*(np.arange(i) for i in chunkshape), indexing="ij")
    for idx in zip(*(i.ravel() for i in grid)):
        print(idx)
        # if os.path.exists(inter_fp(out_fp, idx)):
        #     continue
        tiff_to_zarrs_ijk(in_fp, out_fp, idx)


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
    mmap = tifffile.memmap(in_fp)
    shape = np.array(mmap.shape)
    del mmap
    # Getting the array shape of the "larger" zarr folder chunks
    chunkshape = [int(np.ceil(i / j)) for i, j in zip(shape, INTER_CHUNKS)]
    # Making chunked tiff to intermediate zarr files
    make_inter_zarrs(in_fp, out_fp, chunkshape)
    # Getting array of intermediate zarr files
    arr_c = get_inter_zarrs(out_fp, chunkshape)
    # Combining intermediate zarr files
    arr = da.block(arr_c.tolist())
    arr = arr.rechunk(PROC_CHUNKS)
    arr.to_zarr(out_fp, overwrite=True)
    # Removing intermediate zarr files
    remove_inter_zarrs(out_fp, chunkshape)


if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # # Making Dask cluster and client (thread-based cluster)
    # cluster = LocalCluster(processes=False)
    # client = Client(cluster)
    # print(client.dashboard_link)

    # large_tiff_to_zarr(in_fp, os.path.join(out_dir, "raw.zarr"))

    mmap = tifffile.memmap(in_fp)
    z_f = zarr.open(
        os.path.join(out_dir, "raw.zarr"),
        mode="w",
        shape=mmap.shape,
        dtype=mmap.dtype,
        chunks=PROC_CHUNKS,
    )
    z_f[:] = mmap

    # # Closing client
    # client.close()
