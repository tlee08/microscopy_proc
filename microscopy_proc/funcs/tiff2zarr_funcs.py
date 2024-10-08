import dask
import dask.array as da
import nibabel as nib
import numpy as np
import tifffile
import zarr

# from prefect import task
from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.utils.io_utils import silentremove


def read_tiff(fp):
    ar = tifffile.imread(fp)
    for i in np.arange(len(ar.shape)):
        ar = np.squeeze(ar)
    return ar


# @task
def btiff2zarr(in_fp, out_fp, chunks=PROC_CHUNKS):
    # To intermediate tiff
    mmap_ar = tifffile.memmap(in_fp)
    zarr_ar = zarr.open(
        f"{out_fp}_tmp.zarr",
        mode="w",
        shape=mmap_ar.shape,
        dtype=mmap_ar.dtype,
        chunks=chunks,
    )
    zarr_ar[:] = mmap_ar
    # To final dask tiff
    zarr_ar = da.from_zarr(f"{out_fp}_tmp.zarr")
    zarr_ar.to_zarr(out_fp, overwrite=True)
    # Remove intermediate
    silentremove(f"{out_fp}_tmp.zarr")


# @task
def tiffs2zarr(in_fp_ls, out_fp, chunks=PROC_CHUNKS):
    # Getting shape and dtype
    ar1 = read_tiff(in_fp_ls[0])
    shape = (len(in_fp_ls), *ar1.shape)
    dtype = ar1.dtype
    # Getting list of dask delayed tiffs
    tiffs_ls = [dask.delayed(read_tiff)(i) for i in in_fp_ls]
    # Getting list of dask array tiffs and rechunking each (in prep later rechunking)
    tiffs_ls = [
        da.from_delayed(i, dtype=dtype, shape=shape[1:]).rechunk(chunks[1:])
        for i in tiffs_ls
    ]
    # Stacking tiffs and rechunking
    ar = da.stack(tiffs_ls, axis=0).rechunk(chunks)
    # Saving to zarr
    ar.to_zarr(out_fp, overwrite=True)


# @task
def btiff2niftygz(in_fp, out_fp):
    ar = tifffile.imread(in_fp)
    nib.Nifti1Image(ar, None).to_filename(out_fp)


def read_niftygz(fp):
    img = nib.load(fp)
    return np.array(img.dataobj)
