import dask.array as da
import numpy as np
import pandas as pd
import tifffile

from microscopy_proc.utils.io_utils import silentremove

#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords_to_points_workers(arr: np.ndarray, coords: pd.DataFrame):
    shape = arr.shape  # noqa: F841
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering
    coords = (
        coords[["z", "y", "x"]]
        .round(0)
        .astype(np.int16)
        .query(
            "z >= 0 and z < @shape[0] and y >= 0 and y < @shape[1] and x >= 0 and x < @shape[2]"
        )
        .values
    )
    # Incrementing the coords in the array
    if coords.shape[0] > 0:
        arr[coords[:, 0], coords[:, 1], coords[:, 2]] += 1
    # Return arr
    return arr


def coords_to_points_start(shape: tuple, arr_out_fp: str) -> da.Array:
    # Initialising spatial array
    arr = np.memmap(
        "temp.dat",
        mode="w+",
        shape=shape,
        dtype=np.uint8,
    )
    return arr


def coords_to_points_end(arr, arr_out_fp):
    # # Saving the subsampled array
    tifffile.imwrite(arr_out_fp, arr)
    # Removing temporary memmap
    silentremove("temp.dat")


#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords_to_points(coords: pd.DataFrame, shape: tuple[int, ...], arr_out_fp: str):
    """
    Converts list of coordinates to spatial array single points.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        arr_out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    arr = coords_to_points_start(shape, arr_out_fp)
    # Adding coords to image
    coords_to_points_workers(arr, coords)
    # Saving the subsampled array
    coords_to_points_end(arr, arr_out_fp)


def coords_to_heatmaps(coords: pd.DataFrame, r, shape, arr_out_fp):
    """
    Converts list of coordinates to spatial array as voxels.
    Overlapping areas accumulate in intensity.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        r: radius of the voxels.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        arr_out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    arr = coords_to_points_start(shape, arr_out_fp)

    # Constructing sphere array mask
    zz, yy, xx = np.ogrid[1 : r * 2, 1 : r * 2, 1 : r * 2]
    circ = np.sqrt((xx - r) ** 2 + (yy - r) ** 2 + (zz - r) ** 2) < r
    # Constructing offset indices
    i = np.arange(-r + 1, r)
    z_ind, y_ind, x_ind = np.meshgrid(i, i, i, indexing="ij")
    # Adding coords to image
    for z, y, x, t in zip(z_ind.ravel(), y_ind.ravel(), x_ind.ravel(), circ.ravel()):
        if t:
            coords_i = coords.copy()
            coords_i["z"] += z
            coords_i["y"] += y
            coords_i["x"] += x
            coords_to_points_workers(arr, coords_i)

    # Saving the subsampled array
    coords_to_points_end(arr, arr_out_fp)


def coords_to_regions(coords, shape, arr_out_fp):
    """
    Converts list of coordinates to spatial array.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, `z`, and `id`.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        arr_out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    arr = coords_to_points_start(shape)

    # Adding coords to image with np.apply_along_axis
    def f(coord):
        # Plotting coord to image. Including only coords within the image's bounds
        if np.all((coord >= 0) & (coord < shape)):
            z, y, x, _id = coord
            arr[z, y, x] = _id

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = coords[["z", "y", "x", "id"]].round(0).astype(np.int16)
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)

    # Saving the subsampled array
    coords_to_points_end(arr, arr_out_fp)
