import dask
import dask.array as da
import numpy as np
import tifffile
from cupyx.scipy.ndimage import zoom

# from scipy.ndimage import zoom
from microscopy_proc.utils.cp_utils import (
    clear_cuda_memory_decorator,
    numpy_2_cupy_decorator,
)


def downsmpl_rough_arr(arr: np.ndarray, z_slice, y_slice, x_slice) -> np.ndarray:
    z_slice = int(np.round(1 / z_slice))
    y_slice = int(np.round(1 / y_slice))
    x_slice = int(np.round(1 / x_slice))
    res = arr[::z_slice, ::y_slice, ::x_slice]
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator()
def downsmpl_fine_arr(arr: np.ndarray, z_slice, y_slice, x_slice) -> np.ndarray:
    res = zoom(arr, (z_slice, y_slice, x_slice))
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
    # Returning img array
    return arr


def slice_img(img_in_fp, img_out_fp, slices):
    """
    Assumes `slices` is given in `(x, y, z)` format.
    Assumes the array is stored in `(z, y, x)` format.
    Also stores the output array in `(z, y, x)` format.
    """
    # Setting up variables
    x_slice, y_slice, z_slice = slices
    # Loading stitched numpy array (entire image)
    arr = np.load(img_in_fp, mmap_mode="r")
    # Slicing array (subsampling)
    s_arr = arr[z_slice, y_slice, x_slice]
    # Saving the subsampled array
    np.save(img_out_fp, s_arr)
    return np.load(img_out_fp, mmap_mode="r")


#####################################################################
#       Stitching slices into 3D arr funcs
#####################################################################


def stitch_load_slice(fp):
    return tifffile.memmap(fp)


def stitch_img(fp_ls, img_out_fp):
    """
    Assumes each image is a z-slice, and is in `(y, x)` format.
    The stitched 3D array is stored in `(z, y, x)` format dimensions.
    """
    # Getting shape and dtype
    img1 = tifffile.imread(fp_ls[0])
    shape = (len(fp_ls), img1.shape[0], img1.shape[1])
    dtype = np.uint16
    # Initialising temporary memmap
    load = dask.delayed(stitch_load_slice)
    res_ls = [load(fp) for fp in fp_ls]
    res = da.concatenate(res_ls, axis=0)
    # Saving to file
    tifffile.imwrite(img_out_fp, res)
    # Making MHD header so it can be read by ImageJ
    # make_npy_header(img_out_fp)
