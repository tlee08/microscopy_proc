import contextlib
import logging

import dask
import dask.array
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client

from microscopy_proc.constants import DEPTH, Coords


def block2coords(func, *args: list) -> dd.DataFrame:
    """
    Applies the `func` to `ar`.
    Expects `func` to convert `ar` to coords df (of sorts).

    Importantly, this offsets the coords in each block.

    Process is:
        - Convert dask arrays to delayed object blocks
        - Perform `func([ar1_blocki, ar2_blocki, ...], *args)` for each block
        - At each block, offset the coords by the block's location in the entire array.
    """

    @dask.delayed
    def func_offsetted(args, z_offset, y_offset, x_offset):
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

    # Getting the first da.Array in the args list
    # to store the offsets of each block in ((z/y/x)_offsets)
    # NOTE: assumes all arrays have the same chunks
    # TODO: assert that all arrays have the same chunks
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


def coords2block(df: dd.DataFrame, block_info: dict) -> dd.DataFrame:
    """
    Converts the coords to a block.
    """
    # Getting block info
    z, y, x = block_info[0]["array-location"]
    # Copying df
    df = df.copy()
    # Offsetting
    df[Coords.Z.value] = df[Coords.Z.value] - z[0]
    df[Coords.Y.value] = df[Coords.Y.value] - y[0]
    df[Coords.X.value] = df[Coords.X.value] - x[0]
    # Returning df
    return df


def disk_cache(ar: da.Array, fp):
    ar.to_zarr(fp, overwrite=True)
    return da.from_zarr(fp)


def da_overlap(ar, d=DEPTH):
    return da.overlap.overlap(ar, depth=d, boundary="reflect").rechunk(
        [i + 2 * d for i in ar.chunksize]
    )


def da_trim(ar, d=DEPTH):
    return ar.map_blocks(
        lambda x: x[d:-d, d:-d, d:-d],
        chunks=[tuple(np.array(i) - d * 2) for i in ar.chunks],
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
            logging.debug(client.dashboard_link)
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
    logging.debug(client.dashboard_link)
    try:
        yield
    finally:
        client.close()
        cluster.close()


def const_iter(x, n):
    """
    Iterates the object, `x`, `n` times.
    """
    for _ in range(n):
        yield x
