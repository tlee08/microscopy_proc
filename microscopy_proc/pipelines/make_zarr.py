import os

from dask.distributed import LocalCluster

from microscopy_proc.constants import PROC_CHUNKS
from microscopy_proc.funcs.io_funcs import btiff_to_zarr, tiffs_to_zarr
from microscopy_proc.utils.dask_utils import (
    cluster_proc_dec,
)


@cluster_proc_dec(lambda: LocalCluster())
def tiff_to_zarr(in_fp, out_dir):
    if os.path.isdir(in_fp):
        tiffs_to_zarr(
            [os.path.join(in_fp, f) for f in os.listdir(in_fp)],
            os.path.join(out_dir, "raw.zarr"),
            chunks=PROC_CHUNKS,
        )
    elif os.path.isfile(in_fp):
        btiff_to_zarr(
            in_fp,
            os.path.join(out_dir, "raw.zarr"),
            chunks=PROC_CHUNKS,
        )
    else:
        raise ValueError("Input file path does not exist.")


if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/example"
    # in_fp = "/home/linux1/Desktop/A-1-1/cropped abcd_larger.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    os.makedirs(out_dir, exist_ok=True)

    tiff_to_zarr(in_fp, out_dir)
