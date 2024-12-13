import contextlib
import logging
from typing import Any, Callable

import dask
import dask.array
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, SpecCluster

from microscopy_proc.constants import DEPTH, Coords
from microscopy_proc.utils.misc_utils import const2iter


def block2coords(func, *args: Any) -> dd.DataFrame:
    """
    Applies the `func` to `arr`.
    Expects `func` to convert `arr` to coords df (of sorts).

    Importantly, this offsets the coords in each block using ONLY
    the chunks of the first da.Array object in `args`.

    All da.Arrays must have the same number of blocks but
    can have different chunk sizes.

    Process is:
        - Convert dask arrays to delayed object blocks
        - Perform `func([arr1_blocki, arr2_blocki, ...], *args)` for each block
        - At each block, offset the coords by the block's location in the entire array.
    """
    # Assertions
    arr_blocks_ls = [i for i in args if isinstance(i, da.Array)]
    # Asserting that there is at least one da.Array (i.e curr_chunks not None)
    assert len(arr_blocks_ls) > 0, "At least one da.Array must be passed."
    # Getting first da.Array
    arr0 = arr_blocks_ls[0]
    # Asserting that all da.Arrays have the same number of blocks
    assert all(
        len(i.to_delayed().ravel()) == len(arr0.to_delayed().ravel())
        for i in arr_blocks_ls
    )
    # Converting chunks tuple[tuple] from chunk sizes to block offsets
    curr_chunks_offsets = [np.cumsum([0, *i[:-1]]) for i in arr0.chunks]
    # Creating the meshgrid of offsets to get offsets for each block in order
    z_offsets, y_offsets, x_offsets = np.meshgrid(*curr_chunks_offsets, indexing="ij")
    # Flattening offsets ndarrays to 1D
    z_offsets = z_offsets.ravel()
    y_offsets = y_offsets.ravel()
    x_offsets = x_offsets.ravel()
    # Getting number of blocks
    n = arr0.to_delayed().ravel().shape[0]
    # Converting dask arrays to list of delayed blocks in args list
    args_blocks = [
        i.to_delayed().ravel() if isinstance(i, da.Array) else list(const2iter(i, n))
        for i in args
    ]
    # Transposing so (block, arg) dimensions.
    args_blocks = [list(i) for i in zip(*args_blocks)]

    # Defining the function that offsets the coords in each block
    # Given the block args and offsets, applies the function to each block
    # and offsets the outputted coords for the block.
    @dask.delayed
    def func_offsetted(args: list, z_offset: int, y_offset: int, x_offset: int):
        df = func(*args)
        df.loc[:, Coords.Z.value] = (
            df[Coords.Z.value] + z_offset if Coords.Z.value in df.columns else z_offset
        )
        df.loc[:, Coords.Y.value] = (
            df[Coords.Y.value] + y_offset if Coords.Y.value in df.columns else y_offset
        )
        df.loc[:, Coords.X.value] = (
            df[Coords.X.value] + x_offset if Coords.X.value in df.columns else x_offset
        )
        return df

    # Applying the function to each block
    return dd.from_delayed(
        [
            func_offsetted(args_block, z_offset, y_offset, x_offset)
            for args_block, z_offset, y_offset, x_offset in zip(
                args_blocks, z_offsets, y_offsets, x_offsets
            )
        ]
    )


def coords2block(df: dd.DataFrame | pd.DataFrame, block_info: dict) -> dd.DataFrame:
    """
    Converts the coords to a block, given the block info (so relevant block offsets are used).

    The block info is from Dask, in the map_blocks function
    (and other relevant functions like map_overlap).
    """
    # Getting block info
    z, y, x = block_info[0]["array-location"]
    # Copying df
    df = df.copy()
    # Offsetting
    df[Coords.Z.value] = df[Coords.Z.value] - z[0]
    df[Coords.Y.value] = df[Coords.Y.value] - y[0]
    df[Coords.X.value] = df[Coords.X.value] - x[0]
    return df


def disk_cache(arr: da.Array, fp):
    arr.to_zarr(fp, overwrite=True)
    return da.from_zarr(fp)


def da_overlap(arr, d=DEPTH):
    return da.overlap.overlap(arr, depth=d, boundary="reflect").rechunk(
        [i + 2 * d for i in arr.chunksize]
    )


def da_trim(arr, d=DEPTH):
    return arr.map_blocks(
        lambda x: x[d:-d, d:-d, d:-d],
        chunks=[tuple(np.array(i) - d * 2) for i in arr.chunks],
    )


def my_configs():
    dask.config.set(
        {
            "distributed.scheduler.active-memory-manager.measure": "managed",
            "distributed.worker.memory.rebalance.measure": "managed",
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": False,
            "distributed.worker.memory.terminate": False,
        }
    )


def cluster_proc_dec(cluster_factory: Callable[[], SpecCluster]):
    """
    `cluster_factory` is a function that returns a Dask cluster.
    This function makes the Dask cluster and client, runs the function,
    then closes the client and cluster.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            cluster = cluster_factory()
            client = Client(cluster)
            logging.debug(client.dashboard_link)
            res = func(*args, **kwargs)
            client.close()
            cluster.close()
            return res

        return wrapper

    return decorator


@contextlib.contextmanager
def cluster_proc_contxt(cluster: SpecCluster):
    """
    Makes a Dask cluster and client, runs the body in the context manager,
    then closes the client and cluster.
    """
    client = Client(cluster)
    logging.debug(client.dashboard_link)
    try:
        yield
    finally:
        client.close()
        cluster.close()
