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

    Importantly, this offsets the coords in each block using ONLY
    the chunks of the first da.Array object in `args`.

    All da.Arrays must have the same number of blocks but
    can have different chunk sizes.

    Process is:
        - Convert dask arrays to delayed object blocks
        - Perform `func([ar1_blocki, ar2_blocki, ...], *args)` for each block
        - At each block, offset the coords by the block's location in the entire array.
    """

    # Getting the first da.Array's chunks
    curr_chunks = None
    for arg in args:
        if isinstance(arg, da.Array):
            # If curr_chunks is None, set as chunks of first da.Array
            curr_chunks = curr_chunks or arg.chunks
        break
    # Asserting that curr_chunks is not None (i.e. there is at least one da.Array)
    assert curr_chunks is not None, "At least one da.Array must be passed."
    # Converting chunks tuple[tuple] from chunk sizes to block offsets
    curr_chunks_offsets = [np.cumsum([0, *i[:-1]]) for i in curr_chunks]
    # Creating the meshgrid of offsets to get offsets for each block in order
    z_offsets, y_offsets, x_offsets = np.meshgrid(*curr_chunks_offsets, indexing="ij")
    # Flattening offsets ndarrays to 1D
    z_offsets = z_offsets.ravel()
    y_offsets = y_offsets.ravel()
    x_offsets = x_offsets.ravel()
    # Getting number of blocks
    n = z_offsets.shape[0]
    # Converting dask arrays to list of delayed blocks in args list
    args_blocks = [
        i.to_delayed().ravel() if isinstance(i, da.Array) else list(const_iter(i, n))
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
