import tifffile
import zarr

from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.utils.io_utils import silentremove


def tiff_to_zarr(in_fp, out_fp, chunks=PROC_CHUNKS):
    # To intermediate tiff
    mmap = tifffile.memmap(in_fp)
    z_f = zarr.open(
        f"{out_fp}_tmp.zarr",
        mode="w",
        shape=mmap.shape,
        dtype=mmap.dtype,
        chunks=chunks,
    )
    z_f[:] = mmap
    # To final dask tiff
    z_f = zarr.open(f"{out_fp}_tmp.zarr")
    z_f.to_zarr(out_fp, overwrite=True)
    # Remove intermediate
    silentremove(f"{out_fp}_tmp.zarr")
