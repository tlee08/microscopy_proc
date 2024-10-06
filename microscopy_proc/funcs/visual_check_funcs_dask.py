import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from prefect import flow, task
from microscopy_proc.constants import PROC_CHUNKS, AnnotColumns, Coords
from microscopy_proc.utils.dask_utils import coords2block


# @task
def make_scatter(df):
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.scatterplot(
        x=df[Coords.X.value], y=df[Coords.Y.value], marker=".", alpha=0.2, s=10, ax=ax
    )
    ax.invert_yaxis()


# @task
def make_img(arr, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, cmap="grey", **kwargs)
    ax.axis("off")


# @task
def cell_counts_plot(df):
    id_counts = df[AnnotColumns.ID.value].value_counts()
    id_counts = id_counts.compute() if isinstance(id_counts, dd.Series) else id_counts
    id_counts = id_counts.sort_values()
    sns.scatterplot(id_counts.values)


#####################################################################
#             Converting coordinates to spatial
#####################################################################


# @task
def coords2points_workers(arr: np.ndarray, coords: pd.DataFrame, block_info=None):
    arr = arr.copy()
    shape = arr.shape  # noqa: F841
    # Offsetting coords with chunk space
    if block_info is not None:
        coords = coords2block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
        .round(0)
        .astype(np.int16)
        .query(
            f"z >= 0 and z < {shape[0]} and y >= 0 and y < {shape[1]} and x >= 0 and x < {shape[2]}"
        )
    )
    # Dask to numpy
    coords = coords.compute() if isinstance(coords, dd.DataFrame) else coords
    # Groupby and counts, so we don't drop duplicates
    coords = (
        coords.groupby([Coords.Z.value, Coords.Y.value, Coords.X.value])
        .size()
        .reset_index(name="counts")
    )
    # Incrementing the coords inCoords.Y.valuee array
    if coords.shape[0] > 0:
        arr[coords[Coords.Z.value], coords[Coords.Y.value], coords[Coords.X.value]] += (
            coords["counts"]
        )
    # Return arr
    return arr


# @task
def coords2sphere_workers(
    arr: np.ndarray, coords: pd.DataFrame, r: int, block_info=None
):
    shape = arr.shape  # noqa: F841
    # Offsetting coords with chunk space
    if block_info is not None:
        coords = coords2block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering for pCoords.Y.valuets within the image + radius padding bounds
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
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
            coords_i[Coords.Z.value] += z
            coords_i[Coords.Y.value] += y
            coords_i[Coords.X.value] += x
            arr = coords2points_workers(arr, coords_i)
    # Return arr
    return arr


#####################################################################
#             Converting coordinates to spatial
#####################################################################


# @flow
def coords2points(coords: pd.DataFrame, shape: tuple[int, ...], arr_out_fp: str):
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
    # arr = arr.map_blocks(
    #     lambda i, block_info=None: coords2points_workers(i, coords, block_info)
    # )
    arr = da.map_blocks(coords2points_workers, arr, coords)
    # Computing and saving
    arr.to_zarr(arr_out_fp, overwrite=True)


# @flow
def coords2heatmaps(coords: pd.DataFrame, r, shape, arr_out_fp):
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
        lambda i, block_info=None: coords2sphere_workers(i, coords, r, block_info)
    )
    # Computing and saving
    arr.to_zarr(arr_out_fp, overwrite=True)


# @flow
def coords2regions(coords, shape, arr_out_fp):
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
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value, AnnotColumns.ID.value]]
        .round(0)
        .astype(np.int16)
    )
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)
