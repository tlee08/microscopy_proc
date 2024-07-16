import cupy as cp
import numpy as np

# from cupyx.scipy import ndimage
from scipy import ndimage

from microscopy_proc.utils.cp_utils import clear_cuda_mem_dec

xp = cp
xdimage = ndimage


def downsmpl_rough_arr(arr: np.ndarray, z_scale, y_scale, x_scale) -> np.ndarray:
    z_scale = int(np.round(1 / z_scale))
    y_scale = int(np.round(1 / y_scale))
    x_scale = int(np.round(1 / x_scale))
    res = arr[::z_scale, ::y_scale, ::x_scale]
    return res


@clear_cuda_mem_dec
def downsmpl_fine_arr(arr: np.ndarray, z_scale, y_scale, x_scale) -> np.ndarray:
    # arr = xp.asarray(arr)
    # res = xdimage.zoom(arr, (z_scale, y_scale, x_scale))
    # return res.get()
    res = xdimage.zoom(arr, (z_scale, y_scale, x_scale))
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
