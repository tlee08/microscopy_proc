import dask
import dask.array as da
import numpy as np
import tifffile
import zarr
from natsort import natsorted
from prefect import task

from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.utils.io_utils import silentremove


@task
def read_tiff(fp):
    arr = tifffile.imread(fp)
    for i in np.arange(len(arr.shape)):
        arr = np.squeeze(arr)
    return arr


@task
def btiff_to_zarr(in_fp, out_fp, chunks=PROC_CHUNKS):
    # To intermediate tiff
    arr_mmap = tifffile.memmap(in_fp)
    arr_zarr = zarr.open(
        f"{out_fp}_tmp.zarr",
        mode="w",
        shape=arr_mmap.shape,
        dtype=arr_mmap.dtype,
        chunks=chunks,
    )
    arr_zarr[:] = arr_mmap
    # To final dask tiff
    arr_zarr = da.from_zarr(f"{out_fp}_tmp.zarr")
    arr_zarr.to_zarr(out_fp, overwrite=True)
    # Remove intermediate
    silentremove(f"{out_fp}_tmp.zarr")


@task
def tiffs_to_zarr(in_fp_ls, out_fp, chunks=PROC_CHUNKS):
    # Natsorting in_fp_ls
    in_fp_ls = natsorted(in_fp_ls)
    # Getting shape and dtype
    arr1 = read_tiff(in_fp_ls[0])
    shape = (len(in_fp_ls), *arr1.shape)
    dtype = arr1.dtype
    # Getting list of dask delayed tiffs
    tiffs_ls = [dask.delayed(read_tiff)(i) for i in in_fp_ls]
    # Getting list of dask array tiffs and rechunking each (in prep later rechunking)
    tiffs_ls = [
        da.from_delayed(i, dtype=dtype, shape=shape[1:]).rechunk(chunks[1:])
        for i in tiffs_ls
    ]
    # Stacking tiffs and rechunking
    arr = da.stack(tiffs_ls, axis=0).rechunk(chunks)
    # Saving to zarr
    arr.to_zarr(out_fp, overwrite=True)
