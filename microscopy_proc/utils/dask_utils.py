import contextlib

import dask
import dask.array
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client

from microscopy_proc.constants import DEPTH


def block_to_coords(func, *args: list) -> dd.DataFrame:
    """
    Applies the `func` to `arr`.
    Expects `func` to convert `arr` to coords df (of sorts).

    Importantly, this offsets the coords in each block.

    Process is:
        - Convert dask arrays to delayed object blocks
        - Perform `func([arr1_blocki, arr2_blocki, ...], *args)` for each block
        - At each block, offset the coords by the block's location in the entire array.
    """

    @dask.delayed
    def func_offsetted(args, z_offset, y_offset, x_offset):
        df = func(*args)
        df.loc[:, "z"] = df["z"] + z_offset if "z" in df.columns else z_offset
        df.loc[:, "y"] = df["y"] + y_offset if "y" in df.columns else y_offset
        df.loc[:, "x"] = df["x"] + x_offset if "x" in df.columns else x_offset
        return df

    # Getting the first block in the args list
    # NOTE: assumes all arrays have the same chunks
    z_offsets, y_offsets, x_offsets = ([0], [0], [0])
    for arg in args:
        if isinstance(arg, da.Array):
            # Getting array of [z, y, x] offsets for each chunk
            z_offsets, y_offsets, x_offsets = np.meshgrid(
                *[np.cumsum([0, *i[:-1]]) for i in arg.chunks], indexing="ij"
            )
            z_offsets = z_offsets.ravel()
            y_offsets = y_offsets.ravel()
            x_offsets = x_offsets.ravel()
            break
    n = z_offsets.shape[0]
    # Converting dask arrays to list of delayed blocks in args list
    args = [
        i.to_delayed().ravel() if isinstance(i, da.Array) else const_iter(i, n)
        for i in args
    ]
    # Applying the function to each block
    return dd.from_delayed(
        [
            func_offsetted(args_ls, i, j, k)
            for *args_ls, i, j, k in zip(*args, z_offsets, y_offsets, x_offsets)
        ]
    )


def coords_to_block(df: dd.DataFrame, block_info: dict) -> dd.DataFrame:
    """
    Converts the coords to a block.
    """
    # Getting block info
    z, y, x = block_info[0]["array-location"]
    # Copying df
    df = df.copy()
    # Offsetting
    df["z"] = df["z"] - z[0]
    df["y"] = df["y"] - y[0]
    df["x"] = df["x"] - x[0]
    # Returning df
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


def cluster_proc_dec(cluster_factory):
    """
    `cluster_factory` is a function that returns a cluster.
    Makes a Dask cluster and client, runs the function,
    then closes the client and cluster.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            cluster = cluster_factory()
            client = Client(cluster)
            print(client.dashboard_link)
            res = func(*args, **kwargs)
            client.close()
            cluster.close()
            return res

        return wrapper

    return decorator


@contextlib.contextmanager
def cluster_proc_contxt(cluster):
    """
    Makes a Dask cluster and client, runs the body in the context manager,
    then closes the client and cluster.
    """
    client = Client(cluster)
    print(client.dashboard_link)
    try:
        yield
    finally:
        client.close()
        cluster.close()


def const_iter(arr, n):
    """
    Iterates over the array `arr` `n` times.
    """
    for _ in range(n):
        yield arr
