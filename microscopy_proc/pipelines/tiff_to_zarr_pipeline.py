import os

import tifffile
import zarr

from microscopy_proc.constants import PROC_CHUNKS


def tiff_to_zarr(in_fp, out_fp, chunks=PROC_CHUNKS):
    mmap = tifffile.memmap(in_fp)
    z_f = zarr.open(
        out_fp,
        mode="w",
        shape=mmap.shape,
        dtype=mmap.dtype,
        chunks=chunks,
    )
    z_f[:] = mmap


if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    tiff_to_zarr(in_fp, os.path.join(out_dir, "raw.zarr"))
