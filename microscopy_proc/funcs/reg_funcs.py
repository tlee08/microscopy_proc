import dask
import dask.array as da
import numpy as np
import tifffile
from cupyx.scipy import ndimage as cp_ndimage
import zarr
from natsort import natsorted
from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.utils.io_utils import silentremove
# from scipy.ndimage import zoom
from microscopy_proc.utils.cp_utils import (
    clear_cuda_memory_decorator,
    np_2_cp_decorator,
)


def downsmpl_rough_arr(arr: np.ndarray, z_slice, y_slice, x_slice) -> np.ndarray:
    z_slice = int(np.round(1 / z_slice))
    y_slice = int(np.round(1 / y_slice))
    x_slice = int(np.round(1 / x_slice))
    res = arr[::z_slice, ::y_slice, ::x_slice]
    return res


@clear_cuda_memory_decorator
@np_2_cp_decorator()
def downsmpl_fine_arr(arr: np.ndarray, z_slice, y_slice, x_slice) -> np.ndarray:
    res = cp_ndimage.zoom(arr, (z_slice, y_slice, x_slice))
    return res


def reorient_arr(arr, orient_ls):
    """
    Order of orient_ls is the axis order.
    Negative of an element in orient_ls means that axis is flipped
    Axis order starts from 1, 2, 3, ...

    Example:
        `orient_ls=(-2, 1, 3)` flips the second axis and swaps the first and second axes.
    """
    orient_ls = list(orient_ls)
    # Flipping axes and formatting orient_ls for transposing
    for i in np.arange(len(orient_ls)):
        # ax is the given axes dimension
        ax = orient_ls[i]
        # ax_new is the formatted dimension for transposing (positive and starts from 0)
        ax_new = np.abs(ax) - 1
        orient_ls[i] = ax_new
        # If the given axes dimension is negative, then flip the axis
        if ax < 0:
            arr = np.flip(arr, ax_new)
    # Reordering axes
    arr = arr.transpose(orient_ls)
    # Returning
    return arr


def slice_arr(arr_in_fp, arr_out_fp, slices):
    """
    Assumes `slices` is given in `(x, y, z)` format.
    Assumes the array is stored in `(z, y, x)` format.
    Also stores the output array in `(z, y, x)` format.
    """
    # Setting up variables
    x_slice, y_slice, z_slice = slices
    # Loading stitched numpy array (entire image)
    arr = np.load(arr_in_fp, mmap_mode="r")
    # Slicing array (subsampling)
    s_arr = arr[z_slice, y_slice, x_slice]
    # Saving the subsampled array
    np.save(arr_out_fp, s_arr)
    return np.load(arr_out_fp, mmap_mode="r")


#####################################################################
#       Stitching slices into 3D arr funcs
#####################################################################



def read_tiff(fp):
    arr = tifffile.imread(fp)
    for i in np.arange(len(arr.shape)):
        arr = np.squeeze(arr)
    return arr


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
