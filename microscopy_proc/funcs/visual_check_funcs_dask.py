import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.utils.dask_utils import coords_to_block, disk_cache


def make_scatter(df):
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.scatterplot(x=df["x"], y=df["y"], marker=".", alpha=0.2, s=10, ax=ax)
    ax.invert_yaxis()


def make_img(arr, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, cmap="grey", **kwargs)
    ax.axis("off")


def cell_counts_plot(df):
    id_counts = df["id"].value_counts()
    id_counts = id_counts.compute() if isinstance(id_counts, dd.Series) else id_counts
    id_counts = id_counts.sort_values()
    sns.scatterplot(id_counts.values)


#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords_to_points_workers(arr: np.ndarray, coords: pd.DataFrame, block_info=None):
    arr = arr.copy()
    shape = arr.shape  # noqa: F841
    # Offsetting coords with chunk space
    if block_info is not None:
        coords = coords_to_block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering
    coords = (
        coords[["z", "y", "x"]]
        .round(0)
        .astype(np.int16)
        .query(
            f"z >= 0 and z < {shape[0]} and y >= 0 and y < {shape[1]} and x >= 0 and x < {shape[2]}"
        )
        .values
    )
    # Dask to numpy
    coords = coords.compute() if isinstance(coords, da.Array) else coords
    # Incrementing the coords in the array
    if coords.shape[0] > 0:
        arr[coords[:, 0], coords[:, 1], coords[:, 2]] += 1
    # Return arr
    return arr


def coords_to_sphere_workers(
    arr: np.ndarray, coords: pd.DataFrame, r: int, block_info=None
):
    shape = arr.shape  # noqa: F841
    # Offsetting coords with chunk space
    if block_info is not None:
        coords = coords_to_block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering for points within the image + radius padding bounds
    coords = (
        coords[["z", "y", "x"]]
        .round(0)
        .astype(np.int16)
        .query(
            f"z > {-1*r} and z < {shape[0] + r} and y > {-1*r} and y < {shape[1]}+{r} and x > -1*{r} and x < {shape[2]}+{r}"
        )
    )
    # Dask to pandas
    coords = coords.compute() if isinstance(coords, dd.DataFrame) else coords
    # Constructing index and sphere mask arrays
    i = np.arange(-r, r + 1)
    z_ind, y_ind, x_ind = np.meshgrid(i, i, i, indexing="ij")
    circ = np.square(z_ind) + np.square(y_ind) + np.square(x_ind) <= np.square(r)
    # Adding coords to image
    for z, y, x, t in zip(z_ind.ravel(), y_ind.ravel(), x_ind.ravel(), circ.ravel()):
        if t:
            coords_i = coords.copy()
            coords_i["z"] += z
            coords_i["y"] += y
            coords_i["x"] += x
            arr = coords_to_points_workers(arr, coords_i)
    # Return arr
    return arr


# def coords_to_points_start(shape: tuple, arr_out_fp: str) -> da.Array:
#     # Initialising spatial array
#     da.zeros(shape, chunks=PROC_CHUNKS, dtype=np.uint8).to_zarr(
#         arr_out_fp, overwrite=True
#     )
#     return da.from_zarr(arr_out_fp)


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
    arr = da.zeros(shape, chunks=PROC_CHUNKS, dtype=np.uint8)
    # Adding coords to image
    arr = arr.map_blocks(
        lambda i, block_info=None: coords_to_points_workers(i, coords, block_info)
    )
    # Computing and saving
    arr.to_zarr(arr_out_fp, overwrite=True)


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
    arr = da.zeros(shape, chunks=PROC_CHUNKS, dtype=np.uint8)
    # Adding coords to image
    arr = arr.map_blocks(
        lambda i, block_info=None: coords_to_sphere_workers(i, coords, r, block_info)
    )
    # Computing and saving
    arr = disk_cache(arr, arr_out_fp)


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
    arr = da.zeros(shape, chunks=PROC_CHUNKS, dtype=np.uint8)

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
