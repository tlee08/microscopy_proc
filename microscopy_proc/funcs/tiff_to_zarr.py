import tifffile
import zarr

from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.utils.io_utils import silentremove


def tiff_to_zarr(in_fp, out_fp, chunks=PROC_CHUNKS):
    # To intermediate tiff
    img_mmap = tifffile.memmap(in_fp)
    img_zarr = zarr.open(
        f"{out_fp}_tmp.zarr",
        mode="w",
        shape=img_mmap.shape,
        dtype=img_mmap.dtype,
        chunks=chunks,
    )
    img_zarr[:] = img_mmap
    # To final dask tiff
    img_zarr = zarr.open(f"{out_fp}_tmp.zarr")
    img_zarr.to_zarr(out_fp, overwrite=True)
    # Remove intermediate
    silentremove(f"{out_fp}_tmp.zarr")
