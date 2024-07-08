import dask.array as da
import tifffile

from microscopy_proc.constants import RAW_CHUNKS


def tiff_to_zarr(in_fp, out_fp):
    """
    Converts a tiff file to a zarr file.

    NOTE: works best with a thread-based cluster.
    """

    return mmap_to_zarr(
        tifffile.memmap(in_fp),
        out_fp,
    )


def mmap_to_zarr(mmap, out_fp):
    """
    Converts a tiff file to a zarr file.

    NOTE: works best with a thread-based cluster.
    """
    # cluster = LocalCluster(processes=False)
    # client = Client(cluster)

    arr_raw = da.from_array(
        mmap,
        chunks=RAW_CHUNKS,
    )
    arr_raw.to_zarr(out_fp, overwrite=True)

    # client.close()

    return out_fp
