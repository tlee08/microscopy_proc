import contextlib
from typing import Optional

import dask
import dask.array
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client

from microscopy_proc.constants import S_DEPTH


def block_to_coords(
    func, arr_ls: list[da.Array], args_ls: Optional[list] = []
) -> dd.DataFrame:
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
    def func_offsetted(block_ls, args, z_offset, y_offset, x_offset):
        df = func(*block_ls, *args)
        df.loc[:, "z"] = df["z"] + z_offset if "z" in df.columns else z_offset
        df.loc[:, "y"] = df["y"] + y_offset if "y" in df.columns else y_offset
        df.loc[:, "x"] = df["x"] + x_offset if "x" in df.columns else x_offset
        return df

    # NOTE: assumes all arrays have the same chunks
    inds = np.meshgrid(
        *[np.cumsum([0, *i[:-1]]) for i in arr_ls[0].chunks], indexing="ij"
    )
    return dd.from_delayed(
        [
            func_offsetted(block_ls, args_ls, i, j, k)
            for *block_ls, i, j, k in zip(
                *[i.to_delayed().ravel() for i in arr_ls],
                inds[0].ravel(),
                inds[1].ravel(),
                inds[2].ravel(),
            )
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


def my_trim(arr, d=S_DEPTH):
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
