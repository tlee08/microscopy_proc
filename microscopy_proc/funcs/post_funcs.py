import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from microscopy_proc.utils.io_utils import make_npy_header, silentremove

# TODO: make with dask


def make_maxima_scatter(df):
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.scatterplot(x=df["x"], y=df["y"], marker=".", alpha=0.2, s=10, ax=ax)
    ax.invert_yaxis()


#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords_to_points(coords: pd.DataFrame, shape: tuple[int, ...], img_out_fp: str):
    """
    Converts list of coordinates to spatial array single points.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        img_out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    img = np.memmap(
        "temp.dat",
        mode="w+",
        shape=shape,
        dtype=np.int16,
    )

    # Adding coords to image with np.apply_along_axis
    def f(coord):
        # Plotting coord to image. Including only coords within the image's bounds
        if np.all((coord >= 0) & (coord < shape)):
            z, y, x = coord
            img[z, y, x] = 1

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = coords[["z", "y", "x"]].round(0).astype(np.int16)
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)
    # Saving the subsampled array
    np.save(img_out_fp, img)
    # Making MHD header so it can be read by ImageJ
    make_npy_header(img_out_fp)
    # Removing temporary memmap
    silentremove("temp.dat")
    # Returning img array
    return np.load(img_out_fp, mmap_mode="r")


def coords_to_heatmaps(coords, r, shape, img_out_fp):
    """
    Converts list of coordinates to spatial array as voxels.
    Overlapping areas accumulate in intensity.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        r: radius of the voxels.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        img_out_fp: The output filename.

    Returns:
        The output image array
    """
    shape = tuple(shape)
    shape_padded = (shape[0] + r * 2, shape[1] + r * 2, shape[2] + r * 2)
    # Initialising spatial array
    img = np.memmap(
        "temp.dat",
        mode="w+",
        shape=shape_padded,
        dtype=np.int16,
    )
    # Constructing sphere array mask
    zz, yy, xx = np.ogrid[1 : r * 2, 1 : r * 2, 1 : r * 2]
    circle_arr = np.sqrt((xx - r) ** 2 + (yy - r) ** 2 + (zz - r) ** 2) < r

    # Adding spherical voxels to the image
    def f(coord):
        z, y, x = coord
        # Getting correct img subspace and masking with circle
        if np.all((coord >= 0) & (coord < shape)):
            img[z + 1 : z + 2 * r, y + 1 : y + 2 * r, x + 1 : x + 2 * r][
                circle_arr
            ] += 1

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = coords[["z", "y", "x"]].round(0).astype(np.int16)
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)
    # Saving the subsampled array
    np.save(img_out_fp, img[r:-r, r:-r, r:-r])
    # Making MHD header so it can be read by ImageJ
    make_npy_header(img_out_fp)
    # Removing temporary memmap
    silentremove("temp.dat")
    # Returning img array
    return np.load(img_out_fp, mmap_mode="r")


def coords_to_cells(coords, shape, img_out_fp):
    """
    Converts list of coordinates to spatial array as voxels.
    Overlapping areas accumulate in intensity.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        r: radius of the voxels.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        img_out_fp: The output filename.

    Returns:
        The output image array
    """
    # TODO: NEED TO FIX THIS FUNCTION - IT IS STALLING SOMETIMES
    shape = tuple(shape)
    r = coords["size"].max()
    shape_padded = (shape[0] + r * 2, shape[1] + r * 2, shape[2] + r * 2)
    # Initialising spatial array
    img = np.memmap(
        "temp.dat",
        mode="w+",
        shape=shape_padded,
        dtype=np.int16,
    )

    # Adding spherical voxels to the image
    def f(coord):
        z, y, x, source, size = coord
        coord = np.array((z, y, x))
        # Constructing sphere array mask
        zz, yy, xx = np.ogrid[1 : size * 2, 1 : size * 2, 1 : size * 2]
        circle_arr = (
            np.sqrt((xx - size) ** 2 + (yy - size) ** 2 + (zz - size) ** 2) < size
        )
        # Getting correct img subspace and masking with circle
        if np.all((coord >= 0) & (coord < shape)):
            z = z + r - size
            y = y + r - size
            x = x + r - size
            img[z + 1 : z + 2 * size, y + 1 : y + 2 * size, x + 1 : x + 2 * size][
                circle_arr
            ] = source

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = coords[["z", "y", "x", "source", "size"]].round(0).astype(np.int16)
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)
    # Saving the subsampled array
    np.save(img_out_fp, img[r:-r, r:-r, r:-r])
    # Making MHD header so it can be read by ImageJ
    make_npy_header(img_out_fp)
    # Removing temporary memmap
    silentremove("temp.dat")
    # Returning img array
    return np.load(img_out_fp, mmap_mode="r")


def coords_to_regions(coords, shape, img_out_fp):
    """
    Converts list of coordinates to spatial array.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, `z`, and `id`.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        img_out_fp: The output filename.

    Returns:
        The output image array
    """
    shape = tuple(shape)
    # Initialising spatial array
    img = np.memmap(
        "temp.dat",
        mode="w+",
        shape=shape,
        dtype=np.int32,
    )

    # Adding coords to image with np.apply_along_axis
    def f(coord):
        # Plotting coord to image. Including only coords within the image's bounds
        if np.all((coord >= 0) & (coord < shape)):
            z, y, x, _id = coord
            img[z, y, x] = _id

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = coords[["z", "y", "x", "id"]].round(0).astype(np.int16)
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)
    # Saving the subsampled array
    np.save(img_out_fp, img)
    # Making MHD header so it can be read by ImageJ
    make_npy_header(img_out_fp)
    # Removing temporary memmap
    silentremove("temp.dat")
    # Returning img array
    return np.load(img_out_fp, mmap_mode="r")
